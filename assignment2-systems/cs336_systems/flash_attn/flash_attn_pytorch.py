import math
import torch


class FlashAttentionPytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        """
        Q, K, V: (..., seq_len, d)  任意前导 batch 维度
        返回 O: (..., seq_len, d)
        同时把 L (logsumexp), Q, K, V, O 存入 ctx 供 backward 使用
        """
        *batch_dims, Nq, d = Q.shape
        Nk = K.shape[-2]
        scale = 1.0 / math.sqrt(d)

        # 选择 tile 大小，至少 16x16，且能整除序列长度（题目保证是2的幂且>=16）
        def pick_tile(n):
            for cand in (128, 64, 32, 16):
                if n % cand == 0:
                    return cand
            return 16

        Bq = pick_tile(Nq)
        Bk = pick_tile(Nk)

        # 把所有 batch 维度 flatten 成一维，方便循环处理
        Q_ = Q.reshape(-1, Nq, d)
        K_ = K.reshape(-1, Nk, d)
        V_ = V.reshape(-1, Nk, d)
        B = Q_.shape[0]

        O = torch.empty_like(Q_)
        L = torch.empty(B, Nq, device=Q.device, dtype=Q.dtype)

        Tq = Nq // Bq
        Tk = Nk // Bk

        for i in range(Tq):
            q_lo, q_hi = i * Bq, (i + 1) * Bq
            Qi = Q_[:, q_lo:q_hi, :]  # (B, Bq, d)

            O_i = torch.zeros(B, Bq, d, device=Q.device, dtype=Q.dtype)
            l_i = torch.zeros(B, Bq, device=Q.device, dtype=Q.dtype)
            m_i = torch.full((B, Bq), float('-inf'), device=Q.device, dtype=Q.dtype)

            for j in range(Tk):
                k_lo, k_hi = j * Bk, (j + 1) * Bk
                Kj = K_[:, k_lo:k_hi, :]  # (B, Bk, d)
                Vj = V_[:, k_lo:k_hi, :]  # (B, Bk, d)

                # 公式4: S_ij = Q_i K_j^T * scale
                Sij = torch.einsum('bqd,bkd->bqk', Qi, Kj) * scale  # (B, Bq, Bk)

                # 公式5: 更新行最大值
                m_ij = Sij.max(dim=-1).values          # (B, Bq)
                m_new = torch.maximum(m_i, m_ij)        # (B, Bq)

                P_ij = torch.exp(Sij - m_new.unsqueeze(-1))  # (B, Bq, Bk)
                alpha = torch.exp(m_i - m_new)                # (B, Bq)

                # 公式6: 更新 l 和 O（在线 rescale 累积）
                l_i = alpha * l_i + P_ij.sum(dim=-1)
                O_i = alpha.unsqueeze(-1) * O_i + torch.einsum('bqk,bkd->bqd', P_ij, Vj)

                m_i = m_new

            O_i = O_i / l_i.unsqueeze(-1)
            # 公式12: L_i = m_i + log(l_i)
            L_i = m_i + torch.log(l_i)

            O[:, q_lo:q_hi, :] = O_i
            L[:, q_lo:q_hi] = L_i

        O = O.reshape(*batch_dims, Nq, d)
        L = L.reshape(*batch_dims, Nq)

        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal

        return O


    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        L, Q, K, V, O = ctx.saved_tensors
        is_causal = ctx.is_causal

        d = Q.shape[-1]
        scale = 1.0 / math.sqrt(d)

        # 重新计算 S = QK^T * scale   (B..., Nq, Nk)
        S = torch.einsum('...qd,...kd->...qk', Q, K) * scale

        if is_causal:
            Nq, Nk = S.shape[-2], S.shape[-1]
            causal_mask = torch.triu(
                torch.ones(Nq, Nk, device=S.device, dtype=torch.bool), diagonal=1
            )
            S = S.masked_fill(causal_mask, float('-inf'))

        # 用保存的 L 直接还原归一化后的概率矩阵 P = exp(S - L)
        # 这正是公式 12 的逆用：L_i = m_i + log(l_i)，
        # 所以 exp(S_ij - L_i) = exp(S_ij - m_i) / l_i = 归一化的 softmax 权重
        P = torch.exp(S - L.unsqueeze(-1))  # (B..., Nq, Nk)

        # dV = P^T @ dO
        dV = torch.einsum('...qk,...qd->...kd', P, grad_out)

        # dP = dO @ V^T
        dP = torch.einsum('...qd,...kd->...qk', grad_out, V)

        # D_i = rowsum(dO_i * O_i)，即对每个 query 行求 dO 与 O 的内积
        D = (grad_out * O).sum(dim=-1, keepdim=True)  # (B..., Nq, 1)

        # dS_ij = P_ij * (dP_ij - D_i)
        dS = P * (dP - D)

        if is_causal:
            dS = dS.masked_fill(causal_mask, 0.0)

        # dQ = dS @ K * scale
        dQ = torch.einsum('...qk,...kd->...qd', dS, K) * scale

        # dK = dS^T @ Q * scale
        dK = torch.einsum('...qk,...qd->...kd', dS, Q) * scale

        return dQ, dK, dV, None