import torch
import torch.nn as nn
import torch.nn.functional as F


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PoswiseFeedForwardNet, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.LeakyReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        # inputs: [b_size x len_q x d_model]
        residual = inputs
        output = self.linear(inputs)
        output = self.dropout(output)
        return self.layer_norm(residual + output)

class Backbone(nn.Module):
    def __init__(self, layers, hidden_size, num_heads, dropout_rate) -> None:
        super(Backbone, self).__init__()
        self.layers = layers
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        self.last_layernorm = torch.nn.LayerNorm(self.hidden_size, eps=1e-8)

        for _ in range(self.layers):
            new_attn_layernorm = torch.nn.LayerNorm(self.hidden_size, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = torch.nn.MultiheadAttention(self.hidden_size, self.num_heads, self.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(self.hidden_size, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PoswiseFeedForwardNet(self.hidden_size, self.hidden_size, self.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

    
    def forward(self, seqs, answer):
        # answer: serve as the mask
        timeline_mask = (answer == 0)
        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim
        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, timeline_mask)
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)
        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)
        return log_feats

class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(CrossAttentionBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_size, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, key_padding_mask):
        # Cross-Attention: q and k,v are from different view
        # Self-Attention: q,k,v are from the same view
        attention, _ = self.attention(query, key, value, key_padding_mask=key_padding_mask) # [B * N * H]
        attention *=  ~key_padding_mask.unsqueeze(-1)
        # Add & Norm
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out
