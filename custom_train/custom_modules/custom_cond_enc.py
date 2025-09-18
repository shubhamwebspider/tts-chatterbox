# modules/cond_enc.py
from dataclasses import dataclass
from typing import Optional, Union
import torch
from torch import nn

# robust import of Perceiver with safe fallback
try:
    from .custom_perceiver import Perceiver
except Exception:
    class Perceiver(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

__all__ = ["T3Cond", "T3CondEnc", "Perceiver"]


@dataclass
class T3Cond:
    clap_emb: Optional[torch.Tensor] = None          # (B, D)
    speaker_emb: Optional[torch.Tensor] = None       # (B, D)
    cond_prompt_speech_tokens: Optional[torch.Tensor] = None  # (B, T)
    cond_prompt_speech_emb: Optional[torch.Tensor] = None     # if precomputed

    # NEW FIELDS FOR MULTILINGUAL + ACCENT
    language_id: Optional[torch.Tensor] = None       # (B,)
    accent_id: Optional[torch.Tensor] = None         # (B,)
    # Optional emotion_adv for advanced conditioning
    emotion_adv: Optional[torch.Tensor] = None       # (B, ...)


class T3CondEnc(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.hp = hp
        cond_dim = hp.n_channels

        self.use_perceiver = getattr(hp, 'use_perceiver_resampler', False)
        if self.use_perceiver:
            self.perceiver = Perceiver(
                pre_attention_query_token=32,
                pre_attention_query_size=cond_dim,
                embedding_dim=cond_dim,
                num_attn_heads=4
            )

        # Language & Accent Embeddings
        self.use_language_id = getattr(hp, 'use_language_id', False)
        self.use_accent_id = getattr(hp, 'use_accent_id', False)

        if self.use_language_id:
            self.language_emb = nn.Embedding(hp.num_languages, cond_dim)
        if self.use_accent_id:
            self.accent_emb = nn.Embedding(hp.num_accents, cond_dim)

        total_cond_inputs = sum([
            self.use_language_id,
            self.use_accent_id,
            getattr(hp, 'encoder_type', 'voice_encoder') == "voice_encoder",
        ])
        if total_cond_inputs > 1:
            self.cond_proj = nn.Linear(cond_dim * total_cond_inputs, cond_dim)
        else:
            self.cond_proj = nn.Identity()

    def forward(self, cond: T3Cond) -> torch.Tensor:
        device = next(self.parameters()).device
        B = cond.speaker_emb.shape[0] if cond.speaker_emb is not None else 1

        conds = []

        if cond.speaker_emb is not None:
            emb = cond.speaker_emb.unsqueeze(1)  # (B,1,D)
            if self.use_perceiver:
                emb = self.perceiver(emb)        # (B,Nq,D)
                emb = emb.mean(dim=1, keepdim=True)
            conds.append(emb)

        if self.use_language_id and cond.language_id is not None:
            lang_emb = self.language_emb(cond.language_id)
            conds.append(lang_emb.unsqueeze(1))

        if self.use_accent_id and cond.accent_id is not None:
            accent_emb = self.accent_emb(cond.accent_id)
            conds.append(accent_emb.unsqueeze(1))

        if len(conds) == 0:
            cond_out = torch.zeros(B, 1, self.hp.n_channels, device=device)
        elif len(conds) == 1:
            cond_out = conds[0]
        else:
            cond_cat = torch.cat(conds, dim=-1)
            cond_out = self.cond_proj(cond_cat)

        return cond_out  # (B, 1, D)