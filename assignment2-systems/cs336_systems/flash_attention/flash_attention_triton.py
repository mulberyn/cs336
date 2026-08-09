import torch
import triton
import triton.language as tl


@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr,
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
):
    """
    FlashAttention-2 forward kernel.
    Each instance handles one batch and one query tile.
    """
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # 1) Load Q tile (once per kernel)
    Q_block_ptr = tl.make_block_ptr(
        base=Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    q = tl.load(Q_block_ptr)  # (Q_TILE_SIZE, D)

    # 2) Initialize local accumulators (float32)
    o_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    m_i = tl.full((Q_TILE_SIZE,), -float('inf'), dtype=tl.float32)

    # 3) Set up K/V block pointers (start at offset 0)
    K_block_ptr = tl.make_block_ptr(
        base=K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    # 4) Loop over key tiles
    k_offset = 0
    while k_offset < N_KEYS:
        # Load K and V tiles
        k = tl.load(K_block_ptr)   # (K_TILE_SIZE, D)
        v = tl.load(V_block_ptr)   # (K_TILE_SIZE, D)

        # Compute S = Q @ K^T * scale
        # q: (Q_TILE_SIZE, D), k: (K_TILE_SIZE, D)
        s = tl.dot(q, tl.trans(k)) * scale   # (Q_TILE_SIZE, K_TILE_SIZE)

        # Row-wise max and softmax numerator
        m_ij = tl.max(s, axis=1)             # (Q_TILE_SIZE,)
        p = tl.exp(s - m_ij[:, None])        # (Q_TILE_SIZE, K_TILE_SIZE)
        l_ij = tl.sum(p, axis=1)             # (Q_TILE_SIZE,)

        # Update statistics
        m_new = tl.maximum(m_i, m_ij)        # (Q_TILE_SIZE,)
        exp_mi = tl.exp(m_i - m_new)
        exp_mij = tl.exp(m_ij - m_new)

        # Compute P @ V (cast P to V's dtype before matmul)
        p_cast = p.to(v.dtype)
        pv = tl.dot(p_cast, v)               # (Q_TILE_SIZE, D)

        # Update O_i and l_i
        o_i = o_i * exp_mi[:, None] + pv * exp_mij[:, None]
        l_i = l_i * exp_mi + l_ij * exp_mij
        m_i = m_new

        # Advance block pointers and offset
        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))
        k_offset += K_TILE_SIZE

    # 5) Final normalization and logsumexp
    o_i = o_i / l_i[:, None]                  # (Q_TILE_SIZE, D)
    L_i = m_i + tl.log(l_i)                   # (Q_TILE_SIZE,)

    # 6) Write outputs (cast O to global memory dtype)
    O_block_ptr = tl.make_block_ptr(
        base=O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    o_i_cast = o_i.to(O_ptr.dtype.element_ty)
    tl.store(O_block_ptr, o_i_cast)

    L_block_ptr = tl.make_block_ptr(
        base=L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    tl.store(L_block_ptr, L_i)


class FlashAttentionTritonFunction(torch.autograd.Function):
    """
    Autograd Function that wraps the Triton kernel for FlashAttention-2 forward pass.
    Backward is not implemented.
    """
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        # Support both 2D (seq_len, head_dim) and 3D (batch, seq_len, head_dim) inputs
        original_shape_Q = Q.shape
        if Q.ndim == 2:
            Q = Q.unsqueeze(0)
            K = K.unsqueeze(0)
            V = V.unsqueeze(0)
        batch_size, N_QUERIES, D = Q.shape
        _, N_KEYS, _ = K.shape

        # Ensure contiguous memory
        Q = Q.contiguous()
        K = K.contiguous()
        V = V.contiguous()

        # Allocate output tensors
        O = torch.empty_like(Q)
        L = torch.empty((batch_size, N_QUERIES), dtype=torch.float32, device=Q.device)

        # Strides
        stride_qb, stride_qq, stride_qd = Q.stride()
        stride_kb, stride_kk, stride_kd = K.stride()
        stride_vb, stride_vk, stride_vd = V.stride()
        stride_ob, stride_oq, stride_od = O.stride()
        stride_lb, stride_lq = L.stride()

        # Tile sizes (must be at least 16)
        Q_TILE_SIZE = 32
        K_TILE_SIZE = 32

        # Grid: (number of query tiles, batch size)
        grid = (N_QUERIES // Q_TILE_SIZE, batch_size)

        # Scale factor: 1 / sqrt(head_dim)
        scale = 1.0 / (D ** 0.5)

        # Launch kernel
        flash_fwd_kernel[grid](
            Q, K, V, O, L,
            stride_qb, stride_qq, stride_qd,
            stride_kb, stride_kk, stride_kd,
            stride_vb, stride_vk, stride_vd,
            stride_ob, stride_oq, stride_od,
            stride_lb, stride_lq,
            N_QUERIES, N_KEYS,
            scale,
            D, Q_TILE_SIZE, K_TILE_SIZE,
        )

        # Restore original shape if input was 2D
        if original_shape_Q.ndim == 2:
            O = O.squeeze(0)
            L = L.squeeze(0)

        # Save for backward (though not used)
        ctx.save_for_backward(Q, K, V, O, L)
        return O

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("Backward pass not implemented for Triton kernel")