import math
import torch
import triton
import triton.language as tl


@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    # 每个 program 处理: 一个 batch index, 一个 query tile
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Q, O, L 的 block pointer: 每个 program 只读写一个 query tile
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    # K, V 的 block pointer: 从第 0 个 key tile 开始，循环中沿 key 维度前进
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    # 加载 Q tile（整个循环期间保持在寄存器/SRAM中）
    Qi = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    Qi = Qi.to(tl.float32)

    # 在线 softmax 累积状态
    m_i = tl.full((Q_TILE_SIZE,), value=float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    acc = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

    q_start = query_tile_index * Q_TILE_SIZE

    # causal 情况下只需要遍历到当前 query tile 结束的位置
    if is_causal:
        num_k_tiles = tl.cdiv(q_start + Q_TILE_SIZE, K_TILE_SIZE)
    else:
        num_k_tiles = tl.cdiv(N_KEYS, K_TILE_SIZE)

    # 单一循环，遍历 key tile j
    for j in range(0, num_k_tiles):
        k_start = j * K_TILE_SIZE

        Kj = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        Vj = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
        Kj = Kj.to(tl.float32)
        Vj = Vj.to(tl.float32)

        # S_ij = Q_i K_j^T * scale
        Sij = tl.dot(Qi, tl.trans(Kj)) * scale  # (Q_TILE_SIZE, K_TILE_SIZE)

        if is_causal:
            q_idx = q_start + tl.arange(0, Q_TILE_SIZE)
            k_idx = k_start + tl.arange(0, K_TILE_SIZE)
            causal_mask = q_idx[:, None] >= k_idx[None, :]
            Sij = tl.where(causal_mask, Sij, float("-inf"))

        # 在线 softmax 更新（公式 5、6）
        m_ij = tl.max(Sij, axis=1)
        m_new = tl.maximum(m_i, m_ij)

        alpha = tl.exp(m_i - m_new)
        Pij = tl.exp(Sij - m_new[:, None])

        l_i = alpha * l_i + tl.sum(Pij, axis=1)
        acc = acc * alpha[:, None] + tl.dot(Pij.to(Vj.dtype), Vj)

        m_i = m_new

        # 循环结束时前进 block pointer
        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))

    # 归一化
    acc = acc / l_i[:, None]
    Li = m_i + tl.log(l_i)

    tl.store(O_block_ptr, acc.to(O_block_ptr.type.element_ty), boundary_check=(0, 1))
    tl.store(L_block_ptr, Li.to(L_block_ptr.type.element_ty), boundary_check=(0,))


def _flash_backward_impl(Q, K, V, O, dO, L, scale: float, is_causal: bool):
    """
    对应 Equation 13-19 的反向传播公式。
    Q, K, V, O, dO: (..., N, d)  任意前导 batch 维度
    L: (..., N)
    """
    Q = Q.to(torch.float32)
    K = K.to(torch.float32)
    V = V.to(torch.float32)
    O = O.to(torch.float32)
    dO = dO.to(torch.float32)
    L = L.to(torch.float32)

    N_q = Q.shape[-2]
    N_k = K.shape[-2]

    # Equation 19: D_i = rowsum(dO_i * O_i)
    D = torch.sum(O * dO, dim=-1)  # (..., N_q)

    # Equation 13: S = QK^T * scale
    S = torch.matmul(Q, K.transpose(-2, -1)) * scale  # (..., N_q, N_k)

    if is_causal:
        q_idx = torch.arange(N_q, device=Q.device)[:, None]
        k_idx = torch.arange(N_k, device=Q.device)[None, :]
        causal_mask = q_idx >= k_idx  # (N_q, N_k)
        S = torch.where(causal_mask, S, torch.full_like(S, float("-inf")))

    # Equation 14: P = exp(S - L)   (直接用保存的 logsumexp 还原归一化后的注意力权重)
    P = torch.exp(S - L.unsqueeze(-1))  # (..., N_q, N_k)

    # Equation 15: dV = P^T @ dO
    dV = torch.matmul(P.transpose(-2, -1), dO)  # (..., N_k, d)

    # Equation 16: dP = dO @ V^T
    dP = torch.matmul(dO, V.transpose(-2, -1))  # (..., N_q, N_k)

    # Equation 17-18: dS = P ◦ (dP - D) ，注意 scale 是乘在 S = QK^T*scale 上的，
    # 所以对 S 求导时也要把 scale 带上（dS/dQ, dS/dK 各含一个 scale 因子）
    dS = P * (dP - D.unsqueeze(-1))  # (..., N_q, N_k)

    if is_causal:
        dS = torch.where(causal_mask, dS, torch.zeros_like(dS))

    # Equation 19: dQ = dS @ K * scale
    dQ = torch.matmul(dS, K) * scale  # (..., N_q, d)

    # dK = dS^T @ Q * scale
    dK = torch.matmul(dS.transpose(-2, -1), Q) * scale  # (..., N_k, d)

    return dQ, dK, dV


# 在模块级别用 torch.compile 包一次，避免每次 backward 调用都重新编译
_flash_backward_compiled = torch.compile(_flash_backward_impl)

class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal=False):
        *batch_dims, Nq, d = q.shape
        Nk = k.shape[-2]

        assert q.is_cuda and k.is_cuda and v.is_cuda, "Triton kernel requires CUDA tensors"

        # flatten 所有 batch 维度
        q_ = q.reshape(-1, Nq, d).contiguous()
        k_ = k.reshape(-1, Nk, d).contiguous()
        v_ = v.reshape(-1, Nk, d).contiguous()
        B = q_.shape[0]

        o_ = torch.empty_like(q_)
        l_ = torch.empty(B, Nq, device=q.device, dtype=torch.float32)

        scale = 1.0 / math.sqrt(d)

        # tile 大小：至少 16x16，可根据 d 调整
        Q_TILE_SIZE = 64 if Nq % 64 == 0 else (32 if Nq % 32 == 0 else 16)
        K_TILE_SIZE = 64 if Nk % 64 == 0 else (32 if Nk % 32 == 0 else 16)

        Tq = triton.cdiv(Nq, Q_TILE_SIZE)

        grid = (Tq, B)

        flash_fwd_kernel[grid](
            q_, k_, v_,
            o_, l_,
            q_.stride(0), q_.stride(1), q_.stride(2),
            k_.stride(0), k_.stride(1), k_.stride(2),
            v_.stride(0), v_.stride(1), v_.stride(2),
            o_.stride(0), o_.stride(1), o_.stride(2),
            l_.stride(0), l_.stride(1),
            Nq, Nk,
            scale,
            D=d,
            Q_TILE_SIZE=Q_TILE_SIZE,
            K_TILE_SIZE=K_TILE_SIZE,
            is_causal=is_causal,
        )

        O = o_.reshape(*batch_dims, Nq, d)
        L = l_.reshape(*batch_dims, Nq)

        ctx.save_for_backward(L, q, k, v, O)
        ctx.is_causal = is_causal

        return O


    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        L, Q, K, V, O = ctx.saved_tensors
        is_causal = ctx.is_causal

        d = Q.shape[-1]
        scale = 1.0 / math.sqrt(d)

        dQ, dK, dV = _flash_backward_compiled(Q, K, V, O, grad_out, L, scale, is_causal)

        dQ = dQ.to(Q.dtype)
        dK = dK.to(K.dtype)
        dV = dV.to(V.dtype)

        return dQ, dK, dV, None