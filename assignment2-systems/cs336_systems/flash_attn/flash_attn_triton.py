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


@triton.jit
def flash_bwd_dkdv_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, D_ptr,
    dK_ptr, dV_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_dob, stride_doq, stride_dod,
    stride_lb, stride_lq,
    stride_db, stride_dq,
    stride_dkb, stride_dkk, stride_dkd,
    stride_dvb, stride_dvk, stride_dvd,
    N_QUERIES, N_KEYS,
    scale,
    D_MODEL: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    # 每个 program 处理: 一个 batch index, 一个 key/value tile (j)
    key_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    k_start = key_tile_index * K_TILE_SIZE

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D_MODEL),
        strides=(stride_kk, stride_kd),
        offsets=(k_start, 0),
        block_shape=(K_TILE_SIZE, D_MODEL),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D_MODEL),
        strides=(stride_vk, stride_vd),
        offsets=(k_start, 0),
        block_shape=(K_TILE_SIZE, D_MODEL),
        order=(1, 0),
    )
    dK_block_ptr = tl.make_block_ptr(
        dK_ptr + batch_index * stride_dkb,
        shape=(N_KEYS, D_MODEL),
        strides=(stride_dkk, stride_dkd),
        offsets=(k_start, 0),
        block_shape=(K_TILE_SIZE, D_MODEL),
        order=(1, 0),
    )
    dV_block_ptr = tl.make_block_ptr(
        dV_ptr + batch_index * stride_dvb,
        shape=(N_KEYS, D_MODEL),
        strides=(stride_dvk, stride_dvd),
        offsets=(k_start, 0),
        block_shape=(K_TILE_SIZE, D_MODEL),
        order=(1, 0),
    )

    Kj = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    Vj = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)

    dKj_acc = tl.zeros((K_TILE_SIZE, D_MODEL), dtype=tl.float32)
    dVj_acc = tl.zeros((K_TILE_SIZE, D_MODEL), dtype=tl.float32)

    # Q, dO, L, D 的 block pointer，从第 0 个 query tile 开始，循环中前进
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D_MODEL),
        strides=(stride_qq, stride_qd),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D_MODEL),
        order=(1, 0),
    )
    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_dob,
        shape=(N_QUERIES, D_MODEL),
        strides=(stride_doq, stride_dod),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D_MODEL),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_db,
        shape=(N_QUERIES,),
        strides=(stride_dq,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    # causal: query tile 起点若严格小于 key tile 起点，则该 tile 全部被 mask，跳过
    if is_causal:
        start_i = k_start // Q_TILE_SIZE
    else:
        start_i = 0
    num_q_tiles = tl.cdiv(N_QUERIES, Q_TILE_SIZE)

    # 把 block pointer 前进到 start_i 对应的位置
    Q_block_ptr = tl.advance(Q_block_ptr, (start_i * Q_TILE_SIZE, 0))
    dO_block_ptr = tl.advance(dO_block_ptr, (start_i * Q_TILE_SIZE, 0))
    L_block_ptr = tl.advance(L_block_ptr, (start_i * Q_TILE_SIZE,))
    D_block_ptr = tl.advance(D_block_ptr, (start_i * Q_TILE_SIZE,))

    for i in range(start_i, num_q_tiles):
        q_start = i * Q_TILE_SIZE

        Qi = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
        dOi = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
        Li = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero").to(tl.float32)
        Di = tl.load(D_block_ptr, boundary_check=(0,), padding_option="zero").to(tl.float32)

        # 重算 S_ij，并用 L_i 直接还原归一化的 P_ij
        Sij = tl.dot(Qi, tl.trans(Kj)) * scale  # (Q_TILE_SIZE, K_TILE_SIZE)

        if is_causal:
            q_idx = q_start + tl.arange(0, Q_TILE_SIZE)
            k_idx = k_start + tl.arange(0, K_TILE_SIZE)
            causal_mask = q_idx[:, None] >= k_idx[None, :]
            Sij = tl.where(causal_mask, Sij, float("-inf"))

        Pij = tl.exp(Sij - Li[:, None])  # (Q_TILE_SIZE, K_TILE_SIZE)

        # dV_j += P_ij^T @ dO_i
        dVj_acc += tl.dot(tl.trans(Pij), dOi)

        # dP_ij = dO_i @ V_j^T
        dPij = tl.dot(dOi, tl.trans(Vj))

        # dS_ij = P_ij * (dP_ij - D_i)
        dSij = Pij * (dPij - Di[:, None])

        if is_causal:
            dSij = tl.where(causal_mask, dSij, 0.0)

        # dK_j += dS_ij^T @ Q_i * scale
        dKj_acc += tl.dot(tl.trans(dSij), Qi) * scale

        # 循环末尾前进 block pointer
        Q_block_ptr = tl.advance(Q_block_ptr, (Q_TILE_SIZE, 0))
        dO_block_ptr = tl.advance(dO_block_ptr, (Q_TILE_SIZE, 0))
        L_block_ptr = tl.advance(L_block_ptr, (Q_TILE_SIZE,))
        D_block_ptr = tl.advance(D_block_ptr, (Q_TILE_SIZE,))

    tl.store(dK_block_ptr, dKj_acc.to(dK_block_ptr.type.element_ty), boundary_check=(0, 1))
    tl.store(dV_block_ptr, dVj_acc.to(dV_block_ptr.type.element_ty), boundary_check=(0, 1))


@triton.jit
def flash_bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, D_ptr,
    dQ_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_dob, stride_doq, stride_dod,
    stride_lb, stride_lq,
    stride_db, stride_dq,
    stride_dqb, stride_dqq, stride_dqd,
    N_QUERIES, N_KEYS,
    scale,
    D_MODEL: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    # 每个 program 处理: 一个 batch index, 一个 query tile (i)
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    q_start = query_tile_index * Q_TILE_SIZE

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D_MODEL),
        strides=(stride_qq, stride_qd),
        offsets=(q_start, 0),
        block_shape=(Q_TILE_SIZE, D_MODEL),
        order=(1, 0),
    )
    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_dob,
        shape=(N_QUERIES, D_MODEL),
        strides=(stride_doq, stride_dod),
        offsets=(q_start, 0),
        block_shape=(Q_TILE_SIZE, D_MODEL),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(q_start,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_db,
        shape=(N_QUERIES,),
        strides=(stride_dq,),
        offsets=(q_start,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    dQ_block_ptr = tl.make_block_ptr(
        dQ_ptr + batch_index * stride_dqb,
        shape=(N_QUERIES, D_MODEL),
        strides=(stride_dqq, stride_dqd),
        offsets=(q_start, 0),
        block_shape=(Q_TILE_SIZE, D_MODEL),
        order=(1, 0),
    )

    Qi = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    dOi = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    Li = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero").to(tl.float32)
    Di = tl.load(D_block_ptr, boundary_check=(0,), padding_option="zero").to(tl.float32)

    dQi_acc = tl.zeros((Q_TILE_SIZE, D_MODEL), dtype=tl.float32)

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D_MODEL),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D_MODEL),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D_MODEL),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D_MODEL),
        order=(1, 0),
    )

    if is_causal:
        num_k_tiles = tl.cdiv(q_start + Q_TILE_SIZE, K_TILE_SIZE)
    else:
        num_k_tiles = tl.cdiv(N_KEYS, K_TILE_SIZE)

    for j in range(0, num_k_tiles):
        k_start = j * K_TILE_SIZE

        Kj = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
        Vj = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)

        Sij = tl.dot(Qi, tl.trans(Kj)) * scale

        if is_causal:
            q_idx = q_start + tl.arange(0, Q_TILE_SIZE)
            k_idx = k_start + tl.arange(0, K_TILE_SIZE)
            causal_mask = q_idx[:, None] >= k_idx[None, :]
            Sij = tl.where(causal_mask, Sij, float("-inf"))

        Pij = tl.exp(Sij - Li[:, None])

        dPij = tl.dot(dOi, tl.trans(Vj))
        dSij = Pij * (dPij - Di[:, None])

        if is_causal:
            dSij = tl.where(causal_mask, dSij, 0.0)

        # dQ_i += dS_ij @ K_j * scale
        dQi_acc += tl.dot(dSij, Kj) * scale

        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))

    tl.store(dQ_block_ptr, dQi_acc.to(dQ_block_ptr.type.element_ty), boundary_check=(0, 1))


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

    staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        L, Q, K, V, O = ctx.saved_tensors
        is_causal = ctx.is_causal

        *batch_dims, Nq, d = Q.shape
        Nk = K.shape[-2]
        scale = 1.0 / math.sqrt(d)

        Q_ = Q.reshape(-1, Nq, d).contiguous()
        K_ = K.reshape(-1, Nk, d).contiguous()
        V_ = V.reshape(-1, Nk, d).contiguous()
        O_ = O.reshape(-1, Nq, d).contiguous()
        L_ = L.reshape(-1, Nq).contiguous()
        dO_ = grad_out.reshape(-1, Nq, d).contiguous()
        B = Q_.shape[0]

        # D_i = rowsum(dO_i * O_i)，逐元素计算，很便宜，直接用 PyTorch
        D_ = (dO_ * O_).sum(dim=-1).contiguous()  # (B, Nq)

        dQ_ = torch.zeros_like(Q_)
        dK_ = torch.zeros_like(K_)
        dV_ = torch.zeros_like(V_)

        Q_TILE_SIZE = 64 if Nq % 64 == 0 else (32 if Nq % 32 == 0 else 16)
        K_TILE_SIZE = 64 if Nk % 64 == 0 else (32 if Nk % 32 == 0 else 16)

        Tq = triton.cdiv(Nq, Q_TILE_SIZE)
        Tk = triton.cdiv(Nk, K_TILE_SIZE)

        # kernel 1: 计算 dK, dV，grid = (Tk, B)
        flash_bwd_dkdv_kernel[(Tk, B)](
            Q_, K_, V_, dO_, L_, D_,
            dK_, dV_,
            Q_.stride(0), Q_.stride(1), Q_.stride(2),
            K_.stride(0), K_.stride(1), K_.stride(2),
            V_.stride(0), V_.stride(1), V_.stride(2),
            dO_.stride(0), dO_.stride(1), dO_.stride(2),
            L_.stride(0), L_.stride(1),
            D_.stride(0), D_.stride(1),
            dK_.stride(0), dK_.stride(1), dK_.stride(2),
            dV_.stride(0), dV_.stride(1), dV_.stride(2),
            Nq, Nk,
            scale,
            D_MODEL=d,
            Q_TILE_SIZE=Q_TILE_SIZE,
            K_TILE_SIZE=K_TILE_SIZE,
            is_causal=is_causal,
        )

        # kernel 2: 计算 dQ，grid = (Tq, B)
        flash_bwd_dq_kernel[(Tq, B)](
            Q_, K_, V_, dO_, L_, D_,
            dQ_,
            Q_.stride(0), Q_.stride(1), Q_.stride(2),
            K_.stride(0), K_.stride(1), K_.stride(2),
            V_.stride(0), V_.stride(1), V_.stride(2),
            dO_.stride(0), dO_.stride(1), dO_.stride(2),
            L_.stride(0), L_.stride(1),
            D_.stride(0), D_.stride(1),
            dQ_.stride(0), dQ_.stride(1), dQ_.stride(2),
            Nq, Nk,
            scale,
            D_MODEL=d,
            Q_TILE_SIZE=Q_TILE_SIZE,
            K_TILE_SIZE=K_TILE_SIZE,
            is_causal=is_causal,
        )

        dQ = dQ_.reshape(*batch_dims, Nq, d)
        dK = dK_.reshape(*batch_dims, Nk, d)
        dV = dV_.reshape(*batch_dims, Nk, d)

        return dQ, dK, dV, None