# cdtab/modules/contrastive_loss.py

import torch
import torch.nn.functional as F

def info_nce_one(q, k_pos, k_neg, tau):
    """
    q:      [C]
    k_pos:  [C]
    k_neg:  [M, C]
    """
    q = F.normalize(q, dim=0)
    k_pos = F.normalize(k_pos, dim=0)

    if k_neg.ndim == 2:
        k_neg = F.normalize(k_neg, dim=1)

    pos_logit = torch.matmul(q, k_pos)    # scalar

    if k_neg.numel() == 0:
        logits = pos_logit[None]
        labels = torch.tensor([0], device=logits.device)
    else:
        neg_logits = torch.matmul(k_neg, q)     # [M]
        logits = torch.cat([pos_logit[None], neg_logits], dim=0)
        labels = torch.tensor([0], device=logits.device)

    return F.cross_entropy(logits[None] / tau, labels)


def compute_contrastive_loss(
    x_num, x_cat_oh, pairs, labels,
    tokenizer_q, encoder_q, proj_q, pred_q,
    tokenizer_k, encoder_k, proj_k,
    tau=0.07
):
    
    # q branch
    t_q = tokenizer_q(x_num, x_cat_oh)[:, 1:, :]
    h_q = encoder_q(t_q)
    h_q = h_q.reshape(h_q.size(0), -1)
    z_q = proj_q(h_q)
    p_q = pred_q(z_q)

    # k branch (momentum)
    with torch.no_grad():
        t_k = tokenizer_k(x_num, x_cat_oh)[:, 1:, :]
        h_k = encoder_k(t_k)
        h_k = h_k.reshape(h_k.size(0), -1)
        z_k = proj_k(h_k)

    # ===== InfoNCE =====
    total_loss = torch.zeros((), device=x_num.device)

    for a_idx, p_idx in pairs:
        cls = labels[a_idx]
        neg_idx = torch.nonzero(labels != cls, as_tuple=False).squeeze(1)

        if neg_idx.numel() == 0:
            continue

        k_neg = z_k[neg_idx]

        q1 = p_q[a_idx]
        q2 = p_q[p_idx]
        k1 = z_k[a_idx]
        k2 = z_k[p_idx]

        loss_ap = info_nce_one(q1, k2, k_neg, tau)
        loss_pa = info_nce_one(q2, k1, k_neg, tau)

        total_loss += (loss_ap + loss_pa)
        total_loss = total_loss / len(pairs)

    return total_loss, z_q, p_q, z_k
