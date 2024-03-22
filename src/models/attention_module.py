import torch
import torch.nn as nn


class AttentionModule(nn.Module):
    def __init__(self, input_dims):
        super().__init__()
        self.input_dims = [dim[0] for dim in input_dims]
        self.num_views = len(input_dims)

        self.queries = nn.ModuleList([nn.Linear(dim, dim) for dim in self.input_dims])
        self.keys = nn.ModuleList([nn.Linear(dim, dim) for dim in self.input_dims])
        self.values = nn.ModuleList([nn.Linear(dim, dim) for dim in self.input_dims])

    def forward(self, inputs):
        device = inputs[0].device
        # batch_size = inputs[0].size(0)

        # Compute attention scores
        queries = torch.stack([q(inp.squeeze(1)) for q, inp in zip(self.queries, inputs)], dim=1).to(device)
        keys = torch.stack([k(inp.squeeze(1)) for k, inp in zip(self.keys, inputs)], dim=1).to(device)
        attention_scores = torch.bmm(queries, keys.transpose(1, 2))
        attention_scores = attention_scores / (torch.tensor(self.input_dims, dtype=torch.float32, device=device) ** 0.5)

        # Compute attention weights
        attention_weights = nn.functional.softmax(attention_scores, dim=-1)  # (batch_size, num_views, num_views)
        attention_weights = attention_weights.unsqueeze(-1)  # (batch_size, num_views, num_views, 1)
        attention_weights = attention_weights.expand(-1, -1, -1, self.input_dims[
            0])  # (batch_size, num_views, num_views, input_dim)
        attention_weights = attention_weights.permute(0, 2, 1, 3)  # (batch_size, num_views, num_views, input_dim)

        # Compute weighted sum
        values = torch.stack([v(inp.squeeze(1)) for v, inp in zip(self.values, inputs)], dim=1).to(device)
        weighted_sum = (attention_weights * values.unsqueeze(2)).sum(
            dim=3)  # (batch_size, num_views, num_views, input_dim)
        weighted_sum = weighted_sum.sum(dim=2)  # (batch_size, num_views, input_dim)

        return weighted_sum