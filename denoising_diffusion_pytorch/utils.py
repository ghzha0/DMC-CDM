import torch

# Mean Pooling
def mean_pooling(output, attention_mask):
    mask_expanded = (~(attention_mask==0)).unsqueeze(-1).expand(output.size()).float()
    sum_embeddings = torch.sum(output * mask_expanded, 1)
    sum_mask = mask_expanded.sum(1)
    return sum_embeddings / sum_mask

# Max Pooling
def max_pooling(output, attention_mask):
    mask_expanded = (~(attention_mask==0)).unsqueeze(-1).expand(output.size()).float()
    output[mask_expanded == 0] = float('-inf')
    return torch.max(output, 1)[0]
