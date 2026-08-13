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
    Bq: tl.constexpr,
    Bk: tl.constexpr,
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
        offsets=(query_tile_index * Bq, 0),
        block_shape=(Bq, D),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Bq, 0),
        block_shape=(Bq, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Bq,),
        block_shape=(Bq,),
        order=(0,),
    )

    # K, V 的 block pointer: 从第 0 个 key tile 开始，循环中沿 key 维度前进
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(Bk, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(Bk, D),
        order=(1, 0),
    )

    # 加载 Q tile（整个循环期间保持在寄存器/SRAM中）
    Qi = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    Qi = Qi.to(tl.float32)

    # 在线 softmax 累积状态
    m_i = tl.full((Bq,), value=float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((Bq,), dtype=tl.float32)
    O_i = tl.zeros((Bq, D), dtype=tl.float32)

    q_start = query_tile_index * Bq

    # causal 情况下只需要遍历到当前 query tile 结束的位置
    if is_causal:
        num_k_tiles = tl.cdiv(q_start + Bq, Bk)
    else:
        num_k_tiles = tl.cdiv(N_KEYS, Bk)

    # 单一循环，遍历 key tile j
    for j in range(0, num_k_tiles):
        k_start = j * Bk

        Kj = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        Vj = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
        Kj = Kj.to(tl.float32)
        Vj = Vj.to(tl.float32)

        # S_ij = Q_i K_j^T * scale
        Sij = tl.dot(Qi, tl.trans(Kj)) * scale  # (Bq, Bk)

        if is_causal:
            q_idx = q_start + tl.arange(0, Bq)
            k_idx = k_start + tl.arange(0, Bk)
            causal_mask = q_idx[:, None] >= k_idx[None, :]
            Sij = tl.where(causal_mask, Sij, float("-inf"))

        # 在线 softmax 更新（公式 5、6）
        m_ij = tl.max(Sij, axis=1)
        m_new = tl.maximum(m_i, m_ij)

        alpha = tl.exp(m_i - m_new)
        Pij = tl.exp(Sij - m_new[:, None])

        l_i = alpha * l_i + tl.sum(Pij, axis=1)
        O_i = O_i * alpha[:, None] + tl.dot(Pij.to(Vj.dtype), Vj)

        m_i = m_new

        # 循环结束时前进 block pointer
        K_block_ptr = tl.advance(K_block_ptr, (Bk, 0))
        V_block_ptr = tl.advance(V_block_ptr, (Bk, 0))

    # 归一化
    O_i = O_i / l_i[:, None]
    Li = m_i + tl.log(l_i)

    tl.store(O_block_ptr, O_i.to(O_block_ptr.type.element_ty), boundary_check=(0, 1))
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
    Bq: tl.constexpr,
    Bk: tl.constexpr,
    is_causal: tl.constexpr,
):
    # 每个 program 处理: 一个 batch index, 一个 key/value tile (j)
    key_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    k_start = key_tile_index * Bk

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D_MODEL),
        strides=(stride_kk, stride_kd),
        offsets=(k_start, 0),
        block_shape=(Bk, D_MODEL),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D_MODEL),
        strides=(stride_vk, stride_vd),
        offsets=(k_start, 0),
        block_shape=(Bk, D_MODEL),
        order=(1, 0),
    )
    dK_block_ptr = tl.make_block_ptr(
        dK_ptr + batch_index * stride_dkb,
        shape=(N_KEYS, D_MODEL),
        strides=(stride_dkk, stride_dkd),
        offsets=(k_start, 0),
        block_shape=(Bk, D_MODEL),
        order=(1, 0),
    )
    dV_block_ptr = tl.make_block_ptr(
        dV_ptr + batch_index * stride_dvb,
        shape=(N_KEYS, D_MODEL),
        strides=(stride_dvk, stride_dvd),
        offsets=(k_start, 0),
        block_shape=(Bk, D_MODEL),
        order=(1, 0),
    )

    Kj = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    Vj = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)

    dKj = tl.zeros((Bk, D_MODEL), dtype=tl.float32)
    dVj = tl.zeros((Bk, D_MODEL), dtype=tl.float32)

    # Q, dO, L, D 的 block pointer，从第 0 个 query tile 开始，循环中前进
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D_MODEL),
        strides=(stride_qq, stride_qd),
        offsets=(0, 0),
        block_shape=(Bq, D_MODEL),
        order=(1, 0),
    )
    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_dob,
        shape=(N_QUERIES, D_MODEL),
        strides=(stride_doq, stride_dod),
        offsets=(0, 0),
        block_shape=(Bq, D_MODEL),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(0,),
        block_shape=(Bq,),
        order=(0,),
    )
    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_db,
        shape=(N_QUERIES,),
        strides=(stride_dq,),
        offsets=(0,),
        block_shape=(Bq,),
        order=(0,),
    )

    # causal: query tile 起点若严格小于 key tile 起点，则该 tile 全部被 mask，跳过
    if is_causal:
        start_i = k_start // Bq
    else:
        start_i = 0
    num_q_tiles = tl.cdiv(N_QUERIES, Bq)

    # 把 block pointer 前进到 start_i 对应的位置
    Q_block_ptr = tl.advance(Q_block_ptr, (start_i * Bq, 0))
    dO_block_ptr = tl.advance(dO_block_ptr, (start_i * Bq, 0))
    L_block_ptr = tl.advance(L_block_ptr, (start_i * Bq,))
    D_block_ptr = tl.advance(D_block_ptr, (start_i * Bq,))

    for i in range(start_i, num_q_tiles):
        q_start = i * Bq

        Qi = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
        dOi = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
        Li = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero").to(tl.float32)
        Di = tl.load(D_block_ptr, boundary_check=(0,), padding_option="zero").to(tl.float32)

        # 重算 S_ij，并用 L_i 直接还原归一化的 P_ij
        Sij = tl.dot(Qi, tl.trans(Kj)) * scale  # (Bq, Bk)

        if is_causal:
            q_idx = q_start + tl.arange(0, Bq)
            k_idx = k_start + tl.arange(0, Bk)
            causal_mask = q_idx[:, None] >= k_idx[None, :]
            Sij = tl.where(causal_mask, Sij, float("-inf"))

        Pij = tl.exp(Sij - Li[:, None])  # (Bq, Bk)

        # dV_j += P_ij^T @ dO_i
        dVj += tl.dot(tl.trans(Pij), dOi)

        # dP_ij = dO_i @ V_j^T
        dPij = tl.dot(dOi, tl.trans(Vj))

        # dS_ij = P_ij * (dP_ij - D_i)
        dSij = Pij * (dPij - Di[:, None])

        if is_causal:
            dSij = tl.where(causal_mask, dSij, 0.0)

        # dK_j += dS_ij^T @ Q_i * scale
        dKj += tl.dot(tl.trans(dSij), Qi) * scale

        # 循环末尾前进 block pointer
        Q_block_ptr = tl.advance(Q_block_ptr, (Bq, 0))
        dO_block_ptr = tl.advance(dO_block_ptr, (Bq, 0))
        L_block_ptr = tl.advance(L_block_ptr, (Bq,))
        D_block_ptr = tl.advance(D_block_ptr, (Bq,))

    tl.store(dK_block_ptr, dKj.to(dK_block_ptr.type.element_ty), boundary_check=(0, 1))
    tl.store(dV_block_ptr, dVj.to(dV_block_ptr.type.element_ty), boundary_check=(0, 1))


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
    Bq: tl.constexpr,
    Bk: tl.constexpr,
    is_causal: tl.constexpr,
):
    # 每个 program 处理: 一个 batch index, 一个 query tile (i)
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    q_start = query_tile_index * Bq

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D_MODEL),
        strides=(stride_qq, stride_qd),
        offsets=(q_start, 0),
        block_shape=(Bq, D_MODEL),
        order=(1, 0),
    )
    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_dob,
        shape=(N_QUERIES, D_MODEL),
        strides=(stride_doq, stride_dod),
        offsets=(q_start, 0),
        block_shape=(Bq, D_MODEL),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(q_start,),
        block_shape=(Bq,),
        order=(0,),
    )
    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_db,
        shape=(N_QUERIES,),
        strides=(stride_dq,),
        offsets=(q_start,),
        block_shape=(Bq,),
        order=(0,),
    )
    dQ_block_ptr = tl.make_block_ptr(
        dQ_ptr + batch_index * stride_dqb,
        shape=(N_QUERIES, D_MODEL),
        strides=(stride_dqq, stride_dqd),
        offsets=(q_start, 0),
        block_shape=(Bq, D_MODEL),
        order=(1, 0),
    )

    Qi = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    dOi = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    Li = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero").to(tl.float32)
    Di = tl.load(D_block_ptr, boundary_check=(0,), padding_option="zero").to(tl.float32)

    dQi = tl.zeros((Bq, D_MODEL), dtype=tl.float32)

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D_MODEL),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(Bk, D_MODEL),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D_MODEL),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(Bk, D_MODEL),
        order=(1, 0),
    )

    if is_causal:
        num_k_tiles = tl.cdiv(q_start + Bq, Bk)
    else:
        num_k_tiles = tl.cdiv(N_KEYS, Bk)

    for j in range(0, num_k_tiles):
        k_start = j * Bk

        Kj = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
        Vj = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)

        Sij = tl.dot(Qi, tl.trans(Kj)) * scale

        if is_causal:
            q_idx = q_start + tl.arange(0, Bq)
            k_idx = k_start + tl.arange(0, Bk)
            causal_mask = q_idx[:, None] >= k_idx[None, :]
            Sij = tl.where(causal_mask, Sij, float("-inf"))

        Pij = tl.exp(Sij - Li[:, None])

        dPij = tl.dot(dOi, tl.trans(Vj))
        dSij = Pij * (dPij - Di[:, None])

        if is_causal:
            dSij = tl.where(causal_mask, dSij, 0.0)

        # dQ_i += dS_ij @ K_j * scale
        dQi += tl.dot(dSij, Kj) * scale

        K_block_ptr = tl.advance(K_block_ptr, (Bk, 0))
        V_block_ptr = tl.advance(V_block_ptr, (Bk, 0))

    tl.store(dQ_block_ptr, dQi.to(dQ_block_ptr.type.element_ty), boundary_check=(0, 1))


class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        *batch_dims, Nq, d = Q.shape
        Nk = K.shape[-2]
        scale = 1.0 / math.sqrt(d)

        assert Q.is_cuda and K.is_cuda and V.is_cuda, "Triton kernel requires CUDA tensors"
        
        def pick_tile(n):
            for cand in (64, 32, 16):
                if n % cand == 0:
                    return cand
            return 16

        Bq = pick_tile(Nq)
        Bk = pick_tile(Nk)

        # flatten 所有 batch 维度
        Q_ = Q.reshape(-1, Nq, d).contiguous()
        K_ = K.reshape(-1, Nk, d).contiguous()
        V_ = V.reshape(-1, Nk, d).contiguous()
        B = Q_.shape[0]

        O_ = torch.empty_like(Q_)
        L_ = torch.empty(B, Nq, device=Q.device, dtype=Q.dtype)
        
        Tq = Nq // Bq

        grid = (Tq, B)

        flash_fwd_kernel[grid](
            Q_, K_, V_,
            O_, L_,
            Q_.stride(0), Q_.stride(1), Q_.stride(2),
            K_.stride(0), K_.stride(1), K_.stride(2),
            V_.stride(0), V_.stride(1), V_.stride(2),
            O_.stride(0), O_.stride(1), O_.stride(2),
            L_.stride(0), L_.stride(1),
            Nq, Nk,
            scale,
            D=d,
            Bq=Bq,
            Bk=Bk,
            is_causal=is_causal,
        )

        O = O_.reshape(*batch_dims, Nq, d)
        L = L_.reshape(*batch_dims, Nq)

        ctx.save_for_backward(L, Q, K, V, O)
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

        def pick_tile(n):
            for cand in (64, 32, 16):
                if n % cand == 0:
                    return cand
            return 16

        Bq = pick_tile(Nq)
        Bk = pick_tile(Nk)

        Tq = triton.cdiv(Nq, Bq)
        Tk = triton.cdiv(Nk, Bk)

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
            Bq=Bq,
            Bk=Bk,
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
            Bq=Bq,
            Bk=Bk,
            is_causal=is_causal,
        )

        dQ = dQ_.reshape(*batch_dims, Nq, d)
        dK = dK_.reshape(*batch_dims, Nk, d)
        dV = dV_.reshape(*batch_dims, Nk, d)

        return dQ, dK, dV, None