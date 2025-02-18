import torch
import math
from .cross_attention import *
from .collabrative_extractor import *
from .semantic_extractor import *
from .utils import *


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class inv_cog(nn.Module):
    def __init__(self, user_num, item_num, config):
        super().__init__()
        self.user_num = user_num
        self.padding_token = item_num
        self.config = config
        
        # Embedding Module
        self.cross_att_1 = CrossAttentionBlock(embed_size=config['cross_embed_size'], heads=config['cross_heads'], dropout=config['cross_dropout_rate'], forward_expansion=1)
        self.cross_att_2 = CrossAttentionBlock(embed_size=config['cross_embed_size'], heads=config['cross_heads'], dropout=config['cross_dropout_rate'], forward_expansion=1)
        self.ce = collabrative_extractor(ce_embed_size=config['ce_embed_size'], ce_dropout=config['ce_dropout'], item_num=item_num)
        self.se = semantic_extractor(dataname=config['dataname'], se_emb_path=config['se_emb_path'], extractor=self.config['semantic_extractor'])
        self.backbone = Backbone(layers=3, hidden_size=config['inv_embed_size'], num_heads=config['backbone_heads'], dropout_rate=config['backbone_dropout'])
        if self.config['semantic_extractor'] == 'llama3':
            semantic_hidden = 4096
        elif self.config['semantic_extractor'] == 'bge-m3':
            semantic_hidden = 1024
        elif self.config['semantic_extractor'] == 'Qwen2-7B-Instruct':
            semantic_hidden = 3584
        elif self.config['semantic_extractor'] == 'MiniCPM-2B-dpo-bf16':
            semantic_hidden = 2304
        
        self.adapter = nn.Sequential(
            nn.Linear(semantic_hidden, 512),
            nn.LeakyReLU(),
            nn.Linear(512, config['inv_embed_size'])
        )

        # Conditional Module:
        self.cross_att_3 = CrossAttentionBlock(embed_size=2 * config['cross_embed_size'], heads=config['cross_heads'], dropout=config['cross_dropout_rate'], forward_expansion=1)
        self.t_embedder = TimestepEmbedder(config['inv_embed_size'])
        self.final_layer = nn.Sequential(
            nn.Linear(3 * config['inv_embed_size'], 2 * config['inv_embed_size']),
            nn.LeakyReLU(),
            nn.Linear(2 * config['inv_embed_size'], 2 * config['inv_embed_size'])
        )

        # Predictor Module:
        self.predictor = nn.Sequential(
            nn.Linear(2 * config['inv_embed_size'], 2 * config['inv_embed_size']),
            nn.LeakyReLU(),
            nn.Linear(2 * config['inv_embed_size'], 2 * config['inv_embed_size']),
            nn.LeakyReLU(),
            nn.Linear(2 * config['inv_embed_size'], 1),
        )
        
        self.emb_dropout = nn.Dropout(config['cross_dropout_rate'])
        self.loss_func = nn.BCELoss(reduction='none')
        self.use_cross_att = True

    def id_item_emb(self, log_seqs):
        return self.ce.get_problem_embedding(log_seqs)

    def llm_item_emb(self, log_seqs):
        return self.se.get_problem_embedding(log_seqs)

    def _get_embedding(self, log_seqs):
        id_seq_emb = self.id_item_emb(log_seqs) # (B * N * CH)
        llm_seq_emb = self.llm_item_emb(log_seqs) # (B * N * SH)
        llm_seq_emb = self.adapter(llm_seq_emb)
        
        if self.config['wosemantic']:
            llm_seq_emb *= 0.

        item_seq_emb = torch.cat([id_seq_emb, llm_seq_emb], dim=-1)
        return item_seq_emb

    def log2feats(self, log_seqs, answers):
        id_seqs = self.id_item_emb(log_seqs)
        id_seqs *= id_seqs.shape[-1] ** 0.5  # QKV/sqrt(D)
        id_seqs = self.emb_dropout(id_seqs) # (B N H)
        id_seqs = id_seqs * answers.unsqueeze(-1)

        llm_seqs = self.llm_item_emb(log_seqs)
        llm_seqs = self.adapter(llm_seqs)
        llm_seqs *= llm_seqs.shape[-1] ** 0.5  # QKV/sqrt(D)
        llm_seqs = self.emb_dropout(llm_seqs) # (B N H)
        llm_seqs = llm_seqs * answers.unsqueeze(-1)

        if self.config['wosemantic']:
            llm_seqs *= 0.

        key_padding_mask = (answers == 0)
        
        if self.use_cross_att:
            cross_id_seqs = self.cross_att_1(llm_seqs, id_seqs, id_seqs, key_padding_mask)
            cross_llm_seqs = self.cross_att_2(id_seqs, llm_seqs, llm_seqs, key_padding_mask)
        else:
            cross_id_seqs = id_seqs
            cross_llm_seqs = llm_seqs
        log_feats = torch.cat([cross_id_seqs, cross_llm_seqs], dim=-1) # (B N 2H)
        return log_feats

    def get_user_embedding(self, log_seqs, answers):
        log_feats = self.log2feats(log_seqs, answers)
        return mean_pooling(log_feats, answers) # (B 2H)

    def forward(self, x_t, t, ctx_log_seqs, ctx_answers, classifer_free_guidance):
        # x_t: (B 2H)
        ctx_feats = self.log2feats(ctx_log_seqs, ctx_answers)

        if classifer_free_guidance:
            ctx_feats *= 0.0

        t = self.t_embedder(t)
        key_padding_mask = (ctx_answers == 0)
        x_t = mean_pooling(self.cross_att_3(x_t.unsqueeze(1).expand(ctx_feats.size()), ctx_feats, ctx_feats, key_padding_mask), ctx_answers)

        return self.final_layer(torch.cat([x_t, t], dim=-1))

    def compute_reconstruction_loss(self, log_feats, seq, answer):
        user_feats = log_feats.unsqueeze(1) # (B,1,H)
        with torch.no_grad():
            prob_embs = self._get_embedding(seq)
        prob_logits = torch.sigmoid(
            self.predictor(user_feats - prob_embs)
        ).squeeze(-1)
        labels = torch.where((answer<=0), 0, 1).float()
        mask = (answer != 0).float()
        loss = torch.sum(self.loss_func(prob_logits, labels)  * mask) / torch.sum(mask)
        return loss

    def compute_encoder_loss(self, seq, answer):
        user_feats = self.get_user_embedding(seq, answer).unsqueeze(1) # (B,1,H)
        prob_embs = self._get_embedding(seq) # (B,N,H)
        prob_logits = torch.sigmoid(
            self.predictor(user_feats - prob_embs)
        ).squeeze(-1)
        # print(torch.mean(prob_logits))
        labels = torch.where((answer<=0), 0, 1).float()
        mask = (answer != 0).float()
        loss = torch.sum(self.loss_func(prob_logits, labels) * mask) / torch.sum(mask)
        return loss
