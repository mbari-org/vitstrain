# vitstrain
# Filename: src/losses.py
# Description: Training loss functions
import torch
import torch.nn as nn
import torch.nn.functional as F


class RebalancedContrastiveLoss(nn.Module):
    """
    Rebalanced Supervised Contrastive Loss for Long-Tailed Datasets.

    Balances the feature space by adjusting pulling/pushing forces using class frequencies.
    Higher weights are given to rare tail classes to scale their intra-class compactness.
    """

    def __init__(self, cls_num_list, temperature=0.07):
        super().__init__()
        cls_num_tensor = torch.tensor(cls_num_list, dtype=torch.float32)
        cls_p = cls_num_tensor / cls_num_tensor.sum()

        # Normalized weight: tail classes get higher multipliers to tighten their embeddings
        class_weights = 1.0 / (cls_p + 1e-8)
        class_weights = class_weights / class_weights.max()
        self.register_buffer("class_weights", class_weights)

        self.temperature = temperature

    def forward(self, features, labels):
        """
        Args:
            features: hidden vector of shape [batch_size, hidden_dim]
            labels: ground truth labels of shape [batch_size]
        Returns:
            A scalar loss value.
        """
        device = features.device
        class_weights = self.class_weights.to(device)

        # 1. Normalize features to sit on a hypersphere
        features = F.normalize(features, p=2, dim=1)

        # 2. Compute the similarity matrix (batch_size x batch_size)
        similarity_matrix = torch.matmul(features, features.T)

        # 3. Create a mask to locate positive pairs (same class)
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != features.shape[0]:
            raise ValueError("Num of labels must match num of features")
        mask = torch.eq(labels, labels.T).float().to(device)

        # Mask out self-contrast (diagonal entries)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(features.shape[0], device=device).view(-1, 1),
            0,
        )
        mask = mask * logits_mask

        # 4. Apply rebalancing weights based on the anchors' classes
        batch_weights = class_weights[labels.squeeze(-1)]

        # 5. Compute logits scaled by temperature and class balancing weight multipliers
        scaled_similarity = (similarity_matrix / self.temperature) * batch_weights.unsqueeze(1)

        # For numerical stability, subtract max logit
        logits_max, _ = torch.max(scaled_similarity, dim=1, keepdim=True)
        logits = scaled_similarity - logits_max.detach()

        # 6. Compute log-softmax over the batch forces
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        # 7. Compute the mean of log-likelihood over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)

        # 8. Compute final loss (negative log-likelihood)
        loss = -mean_log_prob_pos
        return loss.mean()
