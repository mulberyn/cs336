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

        *batch_dims, Nq, d = Q.shape
        Nk = K.shape[-2]
        scale = 1.0 / math.sqrt(d)

        def pick_tile(n):
            for cand in (128, 64, 32, 16):
                if n % cand == 0:
                    return cand
            return 16

        Bq = pick_tile(Nq)
        Bk = pick_tile(Nk)
        Tq = Nq // Bq
        Tk = Nk // Bk

        # flatten 所有 batch 维度
        Q_ = Q.reshape(-1, Nq, d)
        K_ = K.reshape(-1, Nk, d)
        V_ = V.reshape(-1, Nk, d)
        O_ = O.reshape(-1, Nq, d)
        L_ = L.reshape(-1, Nq)
        dO_ = grad_out.reshape(-1, Nq, d)
        B = Q_.shape[0]

        dQ = torch.zeros_like(Q_)
        dK = torch.zeros_like(K_)
        dV = torch.zeros_like(V_)

        # 预计算每行 D_i = rowsum(dO_i * O_i)，(B, Nq)
        D_ = (dO_ * O_).sum(dim=-1)

        # 外层循环 K/V tile（j），内层循环 Q tile（i） —— 对应 Algorithm 2
        for j in range(Tk):
            k_lo, k_hi = j * Bk, (j + 1) * Bk
            Kj = K_[:, k_lo:k_hi, :]   # (B, Bk, d)
            Vj = V_[:, k_lo:k_hi, :]   # (B, Bk, d)

            dKj = torch.zeros(B, Bk, d, device=Q.device, dtype=Q.dtype)
            dVj = torch.zeros(B, Bk, d, device=Q.device, dtype=Q.dtype)

            for i in range(Tq):
                q_lo, q_hi = i * Bq, (i + 1) * Bq

                # causal 情况下，若整个 tile 全在对角线上方（未来位置），直接跳过
                if is_causal and k_lo > q_hi - 1:
                    continue

                Qi = Q_[:, q_lo:q_hi, :]     # (B, Bq, d)
                Oi = O_[:, q_lo:q_hi, :]     # (B, Bq, d)  (仅用于对照, D_i已算好)
                Li = L_[:, q_lo:q_hi]        # (B, Bq)
                dOi = dO_[:, q_lo:q_hi, :]   # (B, Bq, d)
                Di = D_[:, q_lo:q_hi]        # (B, Bq)

                # 重算 S_ij 与归一化后的 P_ij = exp(S_ij - L_i)
                Sij = torch.einsum('bqd,bkd->bqk', Qi, Kj) * scale  # (B, Bq, Bk)

                if is_causal:
                    q_idx = torch.arange(q_lo, q_hi, device=Q.device).unsqueeze(-1)
                    k_idx = torch.arange(k_lo, k_hi, device=Q.device).unsqueeze(0)
                    mask = (k_idx > q_idx)  # True 表示未来位置，需要屏蔽
                    Sij = Sij.masked_fill(mask, float('-inf'))

                Pij = torch.exp(Sij - Li.unsqueeze(-1))  # (B, Bq, Bk)

                # dV_j += P_ij^T @ dO_i
                dVj += torch.einsum('bqk,bqd->bkd', Pij, dOi)

                # dP_ij = dO_i @ V_j^T
                dPij = torch.einsum('bqd,bkd->bqk', dOi, Vj)

                # dS_ij = P_ij * (dP_ij - D_i)
                dSij = Pij * (dPij - Di.unsqueeze(-1))

                if is_causal:
                    dSij = dSij.masked_fill(mask, 0.0)

                # dQ_i += dS_ij @ K_j * scale  （累加到全局 dQ）
                dQ[:, q_lo:q_hi, :] += torch.einsum('bqk,bkd->bqd', dSij, Kj) * scale

                # dK_j += dS_ij^T @ Q_i * scale
                dKj += torch.einsum('bqk,bqd->bkd', dSij, Qi) * scale

            dK[:, k_lo:k_hi, :] += dKj
            dV[:, k_lo:k_hi, :] += dVj

        dQ = dQ.reshape(*batch_dims, Nq, d)
        dK = dK.reshape(*batch_dims, Nk, d)
        dV = dV.reshape(*batch_dims, Nk, d)

        return dQ, dK, dV, None