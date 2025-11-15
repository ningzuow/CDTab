import torch

# 假设 q 和 k_pos 都是 3 维向量
q = torch.tensor([1.0, 2.0, 3.0])
k_pos = torch.tensor([4.0, 5.0, 6.0])

# 正样本相似度（标量）
pos_logit = torch.matmul(q, k_pos)

print("pos_logit =", pos_logit)
print("pos_logit shape =", pos_logit.shape)

# 加上 [None]
pos_logit_expanded = pos_logit[None]

print("pos_logit[None] =", pos_logit_expanded)
print("pos_logit[None] shape =", pos_logit_expanded.shape)


# 假设有三个负样本（随机演示）
k_neg = torch.randn(3, 3)
neg_logits = torch.matmul(k_neg, q)

logits = torch.cat([pos_logit[None], neg_logits], dim=0)
print("\nlogits =", logits)
print("logits shape =", logits.shape)

# 再加一层 [None] → CrossEntropy 需要的维度
logits_ce = logits[None]
print("\nlogits[None] =", logits_ce)
print("logits[None] shape =", logits_ce.shape)
