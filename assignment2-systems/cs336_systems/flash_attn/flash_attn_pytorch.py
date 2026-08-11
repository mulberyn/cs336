import torch

class FlashAttentionPytorch(torch.autograd.Function):
    """
    纯 PyTorch 实现的 FlashAttention‑2 前向传播（支持 batch 维度）。
    忽略 is_causal 标志，块大小 Br = Bc = 32（≥16）。
    保存 Q, K, V, O 以及 logsumexp L（形状与 Q 的 batch 维度和序列长度一致）供反向传播使用。
    """
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        # 获取原始形状，支持任意 batch 维度
        original_shape_Q = Q.shape
        if Q.ndim < 2:
            raise ValueError("Q must have at least 2 dimensions")

        # 提取最后两维 (N, d)，其余为 batch 维度
        *batch_dims, N, d = original_shape_Q

        # 展平 batch 维度为第一维
        Q_flat = Q.view(-1, N, d)
        K_flat = K.view(-1, K.shape[-2], K.shape[-1])
        V_flat = V.view(-1, V.shape[-2], V.shape[-1])
        B = Q_flat.shape[0]
        assert K_flat.shape[0] == B and V_flat.shape[0] == B

        O_flat = torch.zeros_like(Q_flat)
        L_flat = torch.zeros(B, N, device=Q.device, dtype=Q.dtype)

        Br = 32
        Bc = 32

        for b in range(B):
            Qb = Q_flat[b]          # (N, d)
            Kb = K_flat[b]
            Vb = V_flat[b]
            M = Kb.shape[0]         # 序列长度（可能与 N 不同）

            Ob = torch.zeros_like(Qb)
            Lb = torch.zeros(N, device=Q.device, dtype=Q.dtype)

            for i in range(0, N, Br):
                Qi = Qb[i:i+Br]
                Br_cur = Qi.shape[0]

                o_i = torch.zeros_like(Qi)
                l_i = torch.zeros(Br_cur, device=Q.device, dtype=Q.dtype)
                m_i = torch.full((Br_cur,), -float('inf'), device=Q.device, dtype=Q.dtype)

                for j in range(0, M, Bc):
                    Kj = Kb[j:j+Bc]
                    Vj = Vb[j:j+Bc]

                    S = torch.matmul(Qi, Kj.T) / (d ** 0.5)
                    m_ij = S.max(dim=1).values
                    P = torch.exp(S - m_ij.unsqueeze(1))
                    l_ij = P.sum(dim=1)

                    m_new = torch.max(m_i, m_ij)
                    exp_mi = torch.exp(m_i - m_new)
                    exp_mij = torch.exp(m_ij - m_new)

                    o_i = o_i * exp_mi.unsqueeze(1) + torch.matmul(P, Vj) * exp_mij.unsqueeze(1)
                    l_i = l_i * exp_mi + l_ij * exp_mij
                    m_i = m_new

                Ob_i = o_i / l_i.unsqueeze(1)
                Lb_i = m_i + torch.log(l_i)

                Ob[i:i+Br] = Ob_i
                Lb[i:i+Br] = Lb_i

            O_flat[b] = Ob
            L_flat[b] = Lb

        # 恢复输出形状
        O = O_flat.view(original_shape_Q)
        # logsumexp 形状：(*batch_dims, N)
        L = L_flat.view(*batch_dims, N)

        # 保存用于反向传播
        ctx.save_for_backward(Q, K, V, O, L)
        return O

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("FlashAttention 反向传播尚未实现")