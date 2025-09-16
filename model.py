
# model.py — Transformer (roster attention) + LSTM (temporal) with minutes & stat heads
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class RosterAttention(nn.Module):
    def __init__(self, emb_dim: int, n_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=n_heads, batch_first=True)
        self.ln = nn.LayerNorm(emb_dim)

    def forward(self, q: torch.Tensor, ctx: torch.Tensor, mask: torch.Tensor):
        """
        q:   [B, 1, D]
        ctx: [B, N, D]
        mask:[B, N]  (1 = keep token, 0 = pad/ignore)
        """
        key_padding_mask = (mask == 0)                   # True = ignore
        B, N = mask.shape
        # Identify batches where ALL tokens are masked
        all_masked = key_padding_mask.all(dim=1)         # [B]

        # Default: copy query as fallback output
        out = q.squeeze(1).clone()                       # [B, D]

        # Run attention only for rows with at least one valid token
        if (~all_masked).any():
            idx = (~all_masked).nonzero(as_tuple=False).squeeze(1)
            q_sub   = q[idx]                              # [B', 1, D]
            ctx_sub = ctx[idx]                            # [B', N, D]
            kpm_sub = key_padding_mask[idx]               # [B', N]
            attn_out, _ = self.attn(q_sub, ctx_sub, ctx_sub, key_padding_mask=kpm_sub)
            out[idx] = attn_out.squeeze(1)

        return self.ln(out)                               # [B, D]

class NBAStatModel(nn.Module):
    def __init__(self,
                 n_players,
                 pregame_dim,
                 hist_dim,
                 emb_dim=1024,
                 lstm_hidden=256,
                 num_layers=8,
                 num_heads=8,
                 ffn_dim=1024,
                 debug=False):
        super().__init__()
        self.debug = debug

        self.player_emb = nn.Embedding(n_players + 1, emb_dim, padding_idx=0)
        self.id_proj = nn.Linear(emb_dim, emb_dim)

        self.pre_proj = nn.Sequential(
            nn.Linear(pregame_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
        )

        self.hist_lstm = nn.LSTM(
            input_size=hist_dim,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True
        )
        self.hist_proj = nn.Linear(lstm_hidden, emb_dim)

        self.roster_attn_team = RosterAttention(emb_dim, n_heads=num_heads)
        self.roster_attn_opp = RosterAttention(emb_dim, n_heads=num_heads)

        self.fuse = nn.Sequential(
            nn.Linear(emb_dim * 4, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
        )

        # Heads
        self.minutes_head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim//2),
            nn.ReLU(),
            nn.Linear(emb_dim//2, 1)
        )
        self.points_head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim//2), nn.ReLU(),
            nn.Linear(emb_dim//2, 1)
        )
        self.reb_head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim//2), nn.ReLU(),
            nn.Linear(emb_dim//2, 1)
        )
        self.ast_head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim//2), nn.ReLU(),
            nn.Linear(emb_dim//2, 1)
        )

    def forward(self, batch):
        B = batch["player_idx"].shape[0]

        # Embeddings
        q = self.player_emb(batch["player_idx"])  # [B,D]
        if self.debug: print("player_emb:", q.shape)
        q = self.id_proj(q).unsqueeze(1)          # [B,1,D]

        team_ctx = self.player_emb(batch["team_ids"])  # [B,N,D]
        opp_ctx  = self.player_emb(batch["opp_ids"])   # [B,N,D]
        if self.debug: 
            print("team_ctx:", team_ctx.shape, "opp_ctx:", opp_ctx.shape)

        # Pregame features
        x_pre = self.pre_proj(batch["pregame"])        # [B,D]
        if self.debug: print("pregame proj:", x_pre.shape)

        # Temporal encoder
        hist_seq = batch["hist_seq"]                   # [B,K,H]
        lstm_out, _ = self.hist_lstm(hist_seq)         # [B,K,Hh]
        if self.debug: print("hist_lstm out:", lstm_out.shape)

        mask = batch["hist_mask"].float().unsqueeze(-1)  
        denom = mask.sum(dim=1).clamp(min=1.0)
        hist_repr = (lstm_out * mask).sum(dim=1) / denom   # [B,Hh]
        hist_repr = self.hist_proj(hist_repr)              # [B,D]
        if self.debug: print("hist_repr:", hist_repr.shape)

        # Roster attention
        team_rep = self.roster_attn_team(q, team_ctx, batch["team_mask"])  
        opp_rep  = self.roster_attn_opp(q, opp_ctx, batch["opp_mask"])     
        if self.debug: 
            print("team_rep:", team_rep.shape, "opp_rep:", opp_rep.shape)

        fused = self.fuse(torch.cat([x_pre, hist_repr, team_rep, opp_rep], dim=-1))  
        if self.debug: print("fused:", fused.shape)

        minutes = 48.0 * torch.sigmoid(self.minutes_head(fused).squeeze(-1))
        points  = F.softplus(self.points_head(fused).squeeze(-1))
        rebs    = F.softplus(self.reb_head(fused).squeeze(-1))
        asts    = F.softplus(self.ast_head(fused).squeeze(-1))

        return {
            "minutes": minutes,
            "points_rate": points,
            "reb_rate": rebs,
            "ast_rate": asts,
            "fused": fused,
        }
