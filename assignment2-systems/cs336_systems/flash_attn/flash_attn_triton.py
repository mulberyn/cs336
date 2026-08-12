import math
import torch
import triton
import triton.language as tl


@triton.jit
def flash_fwd_kernel(
    Q_ptr, Kptr, Vptr,
    Optr, Lptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    KTILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    # 每个 program 处理: 一个 batch index, 一个 query tile
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Q, O, L 的 block pointer: 每个 program 只读写一个 query tile
    Q_blocKptr = tl.make_blocKptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        blocKshape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    OblocKptr = tl.make_blocKptr(
        Optr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        blocKshape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    LblocKptr = tl.make_blocKptr(
        Lptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        blocKshape=(Q_TILE_SIZE,),
        order=(0,),
    )

    # K, V 的 block pointer: 从第 0 个 key tile 开始，循环中沿 key 维度前进
    KblocKptr = tl.make_blocKptr(
        Kptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        blocKshape=(KTILE_SIZE, D),
        order=(1, 0),
    )
    VblocKptr = tl.make_blocKptr(
        Vptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        blocKshape=(KTILE_SIZE, D),
        order=(1, 0),
    )

    # 加载 Q tile（整个循环期间保持在寄存器/SRAM中）
    Qi = tl.load(Q_blocKptr, boundary_check=(0, 1), padding_option="zero")
    Qi = Qi.to(tl.float32)

    # 在线 softmax 累积状态
    m_i = tl.full((Q_TILE_SIZE,), value=float("-inf"), dtype=tl.float32)
    Li = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    acc = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

    Q_start = query_tile_index * Q_TILE_SIZE

    # causal 情况下只需要遍历到当前 query tile 结束的位置
    if is_causal:
        num_Ktiles = tl.cdiv(Q_start + Q_TILE_SIZE, KTILE_SIZE)
    else:
        num_Ktiles = tl.cdiv(N_KEYS, KTILE_SIZE)

    # 单一循环，遍历 key tile j
    for j in range(0, num_Ktiles):
        Kstart = j * KTILE_SIZE

        Kj = tl.load(KblocKptr, boundary_check=(0, 1), padding_option="zero")
        Vj = tl.load(VblocKptr, boundary_check=(0, 1), padding_option="zero")
        Kj = Kj.to(tl.float32)
        Vj = Vj.to(tl.float32)

        # S_ij = Q_i Kj^T * scale
        Sij = tl.dot(Qi, tl.trans(Kj)) * scale  # (Q_TILE_SIZE, KTILE_SIZE)

        if is_causal:
            Q_idx = Q_start + tl.arange(0, Q_TILE_SIZE)
            Kidx = Kstart + tl.arange(0, KTILE_SIZE)
            causaLmask = Q_idx[:, None] >= Kidx[None, :]
            Sij = tl.where(causaLmask, Sij, float("-inf"))

        # 在线 softmax 更新（公式 5、6）
        m_ij = tl.max(Sij, axis=1)
        m_new = tl.maximum(m_i, m_ij)

        alpha = tl.exp(m_i - m_new)
        Pij = tl.exp(Sij - m_new[:, None])

        Li = alpha * Li + tl.sum(Pij, axis=1)
        acc = acc * alpha[:, None] + tl.dot(Pij.to(Vj.dtype), Vj)

        m_i = m_new

        # 循环结束时前进 block pointer
        KblocKptr = tl.advance(KblocKptr, (KTILE_SIZE, 0))
        VblocKptr = tl.advance(VblocKptr, (KTILE_SIZE, 0))

    # 归一化
    acc = acc / Li[:, None]
    Li = m_i + tl.log(Li)

    tl.store(OblocKptr, acc.to(OblocKptr.type.element_ty), boundary_check=(0, 1))
    tl.store(LblocKptr, Li.to(LblocKptr.type.element_ty), boundary_check=(0,))


@triton.jit
def flash_bwd_dkdVkernel(
    Q_ptr, Kptr, Vptr, dOptr, Lptr, D_ptr,
    dKptr, dVptr,
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
    KTILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    # 每个 program 处理: 一个 batch index, 一个 key/value tile (j)
    key_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    Kstart = key_tile_index * KTILE_SIZE

    KblocKptr = tl.make_blocKptr(
        Kptr + batch_index * stride_kb,
        shape=(N_KEYS, D_MODEL),
        strides=(stride_kk, stride_kd),
        offsets=(Kstart, 0),
        blocKshape=(KTILE_SIZE, D_MODEL),
        order=(1, 0),
    )
    VblocKptr = tl.make_blocKptr(
        Vptr + batch_index * stride_vb,
        shape=(N_KEYS, D_MODEL),
        strides=(stride_vk, stride_vd),
        offsets=(Kstart, 0),
        blocKshape=(KTILE_SIZE, D_MODEL),
        order=(1, 0),
    )
    dKblocKptr = tl.make_blocKptr(
        dKptr + batch_index * stride_dkb,
        shape=(N_KEYS, D_MODEL),
        strides=(stride_dkk, stride_dkd),
        offsets=(Kstart, 0),
        blocKshape=(KTILE_SIZE, D_MODEL),
        order=(1, 0),
    )
    dVblocKptr = tl.make_blocKptr(
        dVptr + batch_index * stride_dvb,
        shape=(N_KEYS, D_MODEL),
        strides=(stride_dvk, stride_dvd),
        offsets=(Kstart, 0),
        blocKshape=(KTILE_SIZE, D_MODEL),
        order=(1, 0),
    )

    Kj = tl.load(KblocKptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    Vj = tl.load(VblocKptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)

    dKj_acc = tl.zeros((KTILE_SIZE, D_MODEL), dtype=tl.float32)
    dVj_acc = tl.zeros((KTILE_SIZE, D_MODEL), dtype=tl.float32)

    # Q, dO, L, D 的 block pointer，从第 0 个 query tile 开始，循环中前进
    Q_blocKptr = tl.make_blocKptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D_MODEL),
        strides=(stride_qq, stride_qd),
        offsets=(0, 0),
        blocKshape=(Q_TILE_SIZE, D_MODEL),
        order=(1, 0),
    )
    dOblocKptr = tl.make_blocKptr(
        dOptr + batch_index * stride_dob,
        shape=(N_QUERIES, D_MODEL),
        strides=(stride_doq, stride_dod),
        offsets=(0, 0),
        blocKshape=(Q_TILE_SIZE, D_MODEL),
        order=(1, 0),
    )
    LblocKptr = tl.make_blocKptr(
        Lptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(0,),
        blocKshape=(Q_TILE_SIZE,),
        order=(0,),
    )
    D_blocKptr = tl.make_blocKptr(
        D_ptr + batch_index * stride_db,
        shape=(N_QUERIES,),
        strides=(stride_dq,),
        offsets=(0,),
        blocKshape=(Q_TILE_SIZE,),
        order=(0,),
    )

    # causal: query tile 起点若严格小于 key tile 起点，则该 tile 全部被 mask，跳过
    if is_causal:
        start_i = Kstart // Q_TILE_SIZE
    else:
        start_i = 0
    num_Q_tiles = tl.cdiv(N_QUERIES, Q_TILE_SIZE)

    # 把 block pointer 前进到 start_i 对应的位置
    Q_blocKptr = tl.advance(Q_blocKptr, (start_i * Q_TILE_SIZE, 0))
    dOblocKptr = tl.advance(dOblocKptr, (start_i * Q_TILE_SIZE, 0))
    LblocKptr = tl.advance(LblocKptr, (start_i * Q_TILE_SIZE,))
    D_blocKptr = tl.advance(D_blocKptr, (start_i * Q_TILE_SIZE,))

    for i in range(start_i, num_Q_tiles):
        Q_start = i * Q_TILE_SIZE

        Qi = tl.load(Q_blocKptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
        dOi = tl.load(dOblocKptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
        Li = tl.load(LblocKptr, boundary_check=(0,), padding_option="zero").to(tl.float32)
        Di = tl.load(D_blocKptr, boundary_check=(0,), padding_option="zero").to(tl.float32)

        # 重算 S_ij，并用 Li 直接还原归一化的 P_ij
        Sij = tl.dot(Qi, tl.trans(Kj)) * scale  # (Q_TILE_SIZE, KTILE_SIZE)

        if is_causal:
            Q_idx = Q_start + tl.arange(0, Q_TILE_SIZE)
            Kidx = Kstart + tl.arange(0, KTILE_SIZE)
            causaLmask = Q_idx[:, None] >= Kidx[None, :]
            Sij = tl.where(causaLmask, Sij, float("-inf"))

        Pij = tl.exp(Sij - Li[:, None])  # (Q_TILE_SIZE, KTILE_SIZE)

        # dVj += P_ij^T @ dOi
        dVj_acc += tl.dot(tl.trans(Pij), dOi)

        # dP_ij = dOi @ Vj^T
        dPij = tl.dot(dOi, tl.trans(Vj))

        # dS_ij = P_ij * (dP_ij - D_i)
        dSij = Pij * (dPij - Di[:, None])

        if is_causal:
            dSij = tl.where(causaLmask, dSij, 0.0)

        # dKj += dS_ij^T @ Q_i * scale
        dKj_acc += tl.dot(tl.trans(dSij), Qi) * scale

        # 循环末尾前进 block pointer
        Q_blocKptr = tl.advance(Q_blocKptr, (Q_TILE_SIZE, 0))
        dOblocKptr = tl.advance(dOblocKptr, (Q_TILE_SIZE, 0))
        LblocKptr = tl.advance(LblocKptr, (Q_TILE_SIZE,))
        D_blocKptr = tl.advance(D_blocKptr, (Q_TILE_SIZE,))

    tl.store(dKblocKptr, dKj_acc.to(dKblocKptr.type.element_ty), boundary_check=(0, 1))
    tl.store(dVblocKptr, dVj_acc.to(dVblocKptr.type.element_ty), boundary_check=(0, 1))


@triton.jit
def flash_bwd_dQ_kernel(
    Q_ptr, Kptr, Vptr, dOptr, Lptr, D_ptr,
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
    KTILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    # 每个 program 处理: 一个 batch index, 一个 query tile (i)
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    Q_start = query_tile_index * Q_TILE_SIZE

    Q_blocKptr = tl.make_blocKptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D_MODEL),
        strides=(stride_qq, stride_qd),
        offsets=(Q_start, 0),
        blocKshape=(Q_TILE_SIZE, D_MODEL),
        order=(1, 0),
    )
    dOblocKptr = tl.make_blocKptr(
        dOptr + batch_index * stride_dob,
        shape=(N_QUERIES, D_MODEL),
        strides=(stride_doq, stride_dod),
        offsets=(Q_start, 0),
        blocKshape=(Q_TILE_SIZE, D_MODEL),
        order=(1, 0),
    )
    LblocKptr = tl.make_blocKptr(
        Lptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(Q_start,),
        blocKshape=(Q_TILE_SIZE,),
        order=(0,),
    )
    D_blocKptr = tl.make_blocKptr(
        D_ptr + batch_index * stride_db,
        shape=(N_QUERIES,),
        strides=(stride_dq,),
        offsets=(Q_start,),
        blocKshape=(Q_TILE_SIZE,),
        order=(0,),
    )
    dQ_blocKptr = tl.make_blocKptr(
        dQ_ptr + batch_index * stride_dqb,
        shape=(N_QUERIES, D_MODEL),
        strides=(stride_dqq, stride_dqd),
        offsets=(Q_start, 0),
        blocKshape=(Q_TILE_SIZE, D_MODEL),
        order=(1, 0),
    )

    Qi = tl.load(Q_blocKptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    dOi = tl.load(dOblocKptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    Li = tl.load(LblocKptr, boundary_check=(0,), padding_option="zero").to(tl.float32)
    Di = tl.load(D_blocKptr, boundary_check=(0,), padding_option="zero").to(tl.float32)

    dQi_acc = tl.zeros((Q_TILE_SIZE, D_MODEL), dtype=tl.float32)

    KblocKptr = tl.make_blocKptr(
        Kptr + batch_index * stride_kb,
        shape=(N_KEYS, D_MODEL),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        blocKshape=(KTILE_SIZE, D_MODEL),
        order=(1, 0),
    )
    VblocKptr = tl.make_blocKptr(
        Vptr + batch_index * stride_vb,
        shape=(N_KEYS, D_MODEL),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        blocKshape=(KTILE_SIZE, D_MODEL),
        order=(1, 0),
    )

    if is_causal:
        num_Ktiles = tl.cdiv(Q_start + Q_TILE_SIZE, KTILE_SIZE)
    else:
        num_Ktiles = tl.cdiv(N_KEYS, KTILE_SIZE)

    for j in range(0, num_Ktiles):
        Kstart = j * KTILE_SIZE

        Kj = tl.load(KblocKptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
        Vj = tl.load(VblocKptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)

        Sij = tl.dot(Qi, tl.trans(Kj)) * scale

        if is_causal:
            Q_idx = Q_start + tl.arange(0, Q_TILE_SIZE)
            Kidx = Kstart + tl.arange(0, KTILE_SIZE)
            causaLmask = Q_idx[:, None] >= Kidx[None, :]
            Sij = tl.where(causaLmask, Sij, float("-inf"))

        Pij = tl.exp(Sij - Li[:, None])

        dPij = tl.dot(dOi, tl.trans(Vj))
        dSij = Pij * (dPij - Di[:, None])

        if is_causal:
            dSij = tl.where(causaLmask, dSij, 0.0)

        # dQ_i += dS_ij @ Kj * scale
        dQi_acc += tl.dot(dSij, Kj) * scale

        KblocKptr = tl.advance(KblocKptr, (KTILE_SIZE, 0))
        VblocKptr = tl.advance(VblocKptr, (KTILE_SIZE, 0))

    tl.store(dQ_blocKptr, dQi_acc.to(dQ_blocKptr.type.element_ty), boundary_check=(0, 1))


class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        *batch_dims, Nq, d = Q.shape
        Nk = K.shape[-2]
        scale = 1.0 / math.sqrt(d)

        assert Q.is_cuda and K.is_cuda and V.is_cuda, "Triton kernel requires CUDA tensors"
        
        def pick_tile(n):
            for cand in (128, 64, 32, 16):
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

        O = torch.empty_like(Q_)
        L = torch.empty(B, Nq, device=Q.device, dtype=Q.float32)
        
        Tq = Nq // Bq

        grid = (Tq, B)

        flash_fwd_kernel[grid](
            Q_, K_, V_,
            O, L,
            Q_.stride(0), Q_.stride(1), Q_.stride(2),
            K_.stride(0), K_.stride(1), K_.stride(2),
            V_.stride(0), V_.stride(1), V_.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            Nq, Nk,
            scale,
            D=d,
            Q_TILE_SIZE=Bq,
            KTILE_SIZE=Bk,
            is_causal=is_causal,
        )

        O = O.reshape(*batch_dims, Nq, d)
        L = L.reshape(*batch_dims, Nq)

        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal

        return O

    @staticmethod
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

        # D_i = rowsum(dOi * Oi)，逐元素计算，很便宜，直接用 PyTorch
        D_ = (dO_ * O_).sum(dim=-1).contiguous()  # (B, Nq)

        dQ = torch.zeros_like(Q_)
        dK = torch.zeros_like(K_)
        dV = torch.zeros_like(V_)

        def pick_tile(n):
            for cand in (128, 64, 32, 16):
                if n % cand == 0:
                    return cand
            return 16

        Bq = pick_tile(Nq)
        Bk = pick_tile(Nk)
        
        Tq = Nq // Bq
        Tk = Nk // Bk

        # kernel 1: 计算 dK, dV，grid = (Tk, B)
        flash_bwd_dkdVkernel[(Tk, B)](
            Q_, K_, V_, dO_, L_, D_,
            dK, dV,
            Q_.stride(0), Q_.stride(1), Q_.stride(2),
            K_.stride(0), K_.stride(1), K_.stride(2),
            V_.stride(0), V_.stride(1), V_.stride(2),
            dO_.stride(0), dO_.stride(1), dO_.stride(2),
            L_.stride(0), L_.stride(1),
            D_.stride(0), D_.stride(1),
            dK.stride(0), dK.stride(1), dK.stride(2),
            dV.stride(0), dV.stride(1), dV.stride(2),
            Nq, Nk,
            scale,
            D_MODEL=d,
            Q_TILE_SIZE=Bq,
            KTILE_SIZE=Bk,
            is_causal=is_causal,
        )

        # kernel 2: 计算 dQ，grid = (Tq, B)
        flash_bwd_dQ_kernel[(Tq, B)](
            Q_, K_, V_, dO_, L_, D_,
            dQ,
            Q_.stride(0), Q_.stride(1), Q_.stride(2),
            K_.stride(0), K_.stride(1), K_.stride(2),
            V_.stride(0), V_.stride(1), V_.stride(2),
            dO_.stride(0), dO_.stride(1), dO_.stride(2),
            L_.stride(0), L_.stride(1),
            D_.stride(0), D_.stride(1),
            dQ.stride(0), dQ.stride(1), dQ.stride(2),
            Nq, Nk,
            scale,
            D_MODEL=d,
            Q_TILE_SIZE=Bq,
            KTILE_SIZE=Bk,
            is_causal=is_causal,
        )

        dQ = dQ.reshape(*batch_dims, Nq, d)
        dK = dK.reshape(*batch_dims, Nk, d)
        dV = dV.reshape(*batch_dims, Nk, d)

        return dQ, dK, dV, None