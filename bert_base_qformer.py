# mdse_v3.py — MDSE with BERT-based Q-Former (BLIP-2 style)
# Pipeline: FE -> (SAM masks offline) -> Dual-Pathway (EG ⊕ ES) -> Context Gate -> Q-Former (BERT) -> LLM

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import (
    CLIPModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BertConfig,
    BertModel,  # Q-Former base
)

# ----------------------------
# Config
# ----------------------------
@dataclass
class MDSEConfig:
    # Vision
    clip_name: str = "openai/clip-vit-large-patch14"
    image_size: int = 224
    # Q-Former (BERT-based)
    q_layers: int = 12
    q_heads: int = 12
    q_dim: int = 768
    q_mlp: int = 3072
    num_queries: int = 32
    # LLM
    llm_name: str = "facebook/opt-2.7b"  # or "Salesforce/blip2-opt-2.7b"
    max_gen_len: int = 32
    # Loss weights (λ_ITC, λ_ITM, λ_ITG)
    lambda_itc: float = 1.0
    lambda_itm: float = 0.5
    lambda_itg: float = 0.1
    # LoRA (attach externally with PEFT if desired)
    enable_lora: bool = False
    lora_r_qformer: int = 8
    lora_alpha_qformer: int = 16
    lora_r_llm_proj: int = 8   # projector 768->d_L
    lora_alpha_llm_proj: int = 16
    lora_dropout: float = 0.05


# ----------------------------
# 3.1 Feature Enhancement (compact differentiable version)
# ----------------------------
class FeatureEnhancement(nn.Module):
    def __init__(self):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(0.9))
        self.sharp = nn.Parameter(torch.tensor(0.12))
        self.sat   = nn.Parameter(torch.tensor(1.05))
        self.register_buffer("eps", torch.tensor(1e-6))

    def multi_exposure_fusion(self, x: torch.Tensor) -> torch.Tensor:
        g = [0.7, 1.0, 1.4]
        expos = [torch.clamp(x ** gk, 0, 1) for gk in g]
        mu = 0.5
        ex_w = [torch.exp(-0.5 * ((e - mu) ** 2) / (0.2 ** 2)).mean(dim=1, keepdim=True) for e in expos]
        gray = [e.mean(dim=1, keepdim=True) for e in expos]
        if hasattr(F, "laplacian"):
            lap  = [F.laplacian(gv) for gv in gray]
        else:
            k = torch.tensor([[[[0,-1,0],[-1,4,-1],[0,-1,0]]]], device=x.device, dtype=x.dtype)
            lap  = [F.conv2d(gv, weight=k, padding=1) for gv in gray]
        ct_w = [l.abs() for l in lap]
        W = [ex + ct for ex, ct in zip(ex_w, ct_w)]
        W = [w / (w.mean(dim=(2,3), keepdim=True) + self.eps) for w in W]
        num = sum([w * e for w, e in zip(W, expos)])
        den = sum(W) + self.eps
        return torch.clamp(num / den, 0, 1)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        x = img.clamp(0, 1)
        x = self.multi_exposure_fusion(x)
        x = torch.pow(x + self.eps, self.gamma)
        blur = F.avg_pool2d(x, 3, 1, 1)
        x = torch.clamp((1 + self.sharp) * x - self.sharp * blur, 0, 1)
        mean = x.mean(dim=(2,3), keepdim=True) + self.eps
        gain = mean.mean(dim=1, keepdim=True) / mean
        x = torch.clamp(x * gain, 0, 1)
        maxc, _ = x.max(dim=1, keepdim=True); minc, _ = x.min(dim=1, keepdim=True)
        s = (maxc - minc) / (maxc + self.eps)
        s = torch.clamp(s * self.sat, 0, 1)
        x = (x - minc) * (s / (s + self.eps)) + minc
        return x.clamp(0, 1)


# ----------------------------
# SAM region encoder helper (SAM run offline; here we only consume masks)
# ----------------------------
def encode_regions_with_clip(clip_vision, pixel_values: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """
    pixel_values: [B,3,H,W], masks: [B,K,H,W] in {0,1}
    returns ES pooled embedding: [B, Dv]
    """
    B, K = masks.shape[:2]
    if K == 0:
        Dv = clip_vision.config.hidden_size
        return torch.zeros(B, Dv, device=pixel_values.device, dtype=pixel_values.dtype)

    pv = pixel_values.unsqueeze(1) * masks.unsqueeze(2)  # [B,K,3,H,W]
    pv = pv.reshape(B * K, 3, pixel_values.size(2), pixel_values.size(3))
    out = clip_vision(pixel_values=pv)
    pooled = out.pooler_output  # [B*K, Dv]
    pooled = pooled.reshape(B, K, -1).mean(dim=1)        # [B, Dv]
    return pooled


class DualPathway(nn.Module):
    def __init__(self, vis_dim: int):
        super().__init__()
        self.proj = nn.Linear(vis_dim * 2, vis_dim)
        self.norm = nn.LayerNorm(vis_dim)

    def forward(self, EG: torch.Tensor, ES: torch.Tensor) -> torch.Tensor:
        EF = self.proj(torch.cat([EG, ES], dim=-1))
        return self.norm(EF)  # [B, Dv]


class ContextGate(nn.Module):
    def __init__(self, d_vis: int, d_ctx: int):
        super().__init__()
        self.f_ctx = nn.Linear(d_ctx, d_vis)
        self.g_proj = nn.Linear(d_vis + d_vis, d_vis)

    def forward(self, E_tokens: torch.Tensor, C: Optional[torch.Tensor]) -> torch.Tensor:
        if C is None:
            return E_tokens
        Cprime = self.f_ctx(C).unsqueeze(1).expand_as(E_tokens)
        gate = torch.sigmoid(self.g_proj(torch.cat([E_tokens, Cprime], dim=-1)))
        return E_tokens + gate


class MDSE(nn.Module):
    def __init__(self, cfg: MDSEConfig):
        super().__init__()
        self.cfg = cfg

        # FE + Vision encoder
        self.fe = FeatureEnhancement()
        self.clip = CLIPModel.from_pretrained(cfg.clip_name)
        self.vis_dim = self.clip.vision_model.config.hidden_size  # 1024 for ViT-L/14

        # Dual-path fusion + context gate
        self.dual = DualPathway(self.vis_dim)
        self.ctx_gate = ContextGate(self.vis_dim, d_ctx=self.vis_dim)

        # Project vision tokens to Q-Former width
        self.vis2q = nn.Linear(self.vis_dim, cfg.q_dim)

        # ---- Q-Former (BERT-based with cross-attention) ----
        qconf = BertConfig(
            hidden_size=cfg.q_dim,
            num_hidden_layers=cfg.q_layers,
            num_attention_heads=cfg.q_heads,
            intermediate_size=cfg.q_mlp,
            add_cross_attention=True,   # allow queries to attend visual tokens
            is_decoder=False,           # encoder-style
            encoder_hidden_size=cfg.q_dim,
        )
        self.qformer = BertModel(qconf)
        # Learned queries (1, M, d)
        self.query_tokens = nn.Parameter(torch.randn(1, cfg.num_queries, cfg.q_dim) / math.sqrt(cfg.q_dim))

        # LLM + projector
        self.llm = AutoModelForCausalLM.from_pretrained(cfg.llm_name)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.llm_name)
        d_L = self.llm.config.hidden_size  # 2560 for OPT-2.7B
        self.to_llm = nn.Linear(cfg.q_dim, d_L)

        # Heads for ITM & retrieval
        self.itm_head = nn.Linear(cfg.q_dim, 2)
        self.txt_head = nn.Linear(d_L, d_L, bias=False)
        self.img_head = nn.Linear(d_L, d_L, bias=False)

    @torch.no_grad()
    def _clip_tokens_and_pool(self, pixel_values: torch.Tensor):
        out = self.clip.vision_model(pixel_values=pixel_values)
        tokens = out.last_hidden_state           # [B,T, Dv]
        pooled = out.pooler_output               # [B, Dv]
        return tokens, pooled

    def visual_paths(self, images: torch.Tensor, sam_masks: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        Ihat = self.fe(images)                                    # Î
        tokens, EG = self._clip_tokens_and_pool(Ihat)             # EG
        if sam_masks is not None:
            ES = encode_regions_with_clip(self.clip.vision_model, Ihat, sam_masks)
        else:
            ES = torch.zeros_like(EG)
        return tokens, EG, ES

    # ---- Q-Former forward (queries cross-attend to vision tokens) ----
    def _qformer_forward(self, vis_tokens_qdim: torch.Tensor, vis_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        vis_tokens_qdim: [B, T, d_q]
        vis_mask: [B, T] (1 for valid)
        returns: query outputs [B, M, d_q]
        """
        B = vis_tokens_qdim.size(0)
        q = self.query_tokens.expand(B, -1, -1)  # [B,M,d]
        qmask = torch.ones(B, q.size(1), device=vis_tokens_qdim.device, dtype=torch.long)
        kmask = torch.ones(B, vis_tokens_qdim.size(1), device=vis_tokens_qdim.device, dtype=torch.long) if vis_mask is None else vis_mask
        out = self.qformer(
            inputs_embeds=q,
            attention_mask=qmask,
            encoder_hidden_states=vis_tokens_qdim,
            encoder_attention_mask=kmask,
            return_dict=True,
        )
        return out.last_hidden_state  # [B,M,d]

    def forward_caption(self, images: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                        sam_masks: Optional[torch.Tensor] = None, context_vec: Optional[torch.Tensor] = None):
        tokens_I, EG, ES = self.visual_paths(images, sam_masks)
        EF = self.dual(EG, ES)                                   # [B, Dv]
        # Gate token stream with EF as context
        Ktokens = self.ctx_gate(tokens_I, C=EF if context_vec is None else context_vec)
        Kq = self.vis2q(Ktokens)                                  # [B,T,d_q]
        Qout = self._qformer_forward(Kq)                          # [B,M,d_q]
        vprefix = self.to_llm(Qout)                               # [B,M,d_L]

        # Teacher forcing with visual prefix
        B = images.size(0)
        emb = self.llm.get_input_embeddings()(input_ids)          # [B,L,d_L]
        emb = torch.cat([vprefix, emb], dim=1)
        pfx_mask = torch.ones(B, vprefix.size(1), device=emb.device, dtype=attention_mask.dtype)
        attn = torch.cat([pfx_mask, attention_mask], dim=1)

        out = self.llm(inputs_embeds=emb, attention_mask=attn)
        logits = out.logits                                       # [B,M+L,V]

        # Losses
        L = input_ids.size(1)
        pred = logits[:, -L-1:-1, :]
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        loss_itg = F.cross_entropy(pred.reshape(-1, pred.size(-1)), labels.reshape(-1), ignore_index=-100)

        qavg = Qout.mean(dim=1)
        itm_logits = self.itm_head(qavg)
        targets = torch.ones(B, dtype=torch.long, device=images.device)
        loss_itm = F.cross_entropy(itm_logits, targets)

        with torch.no_grad():
            text_out = self.llm(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            tvec = text_out.hidden_states[-1].mean(dim=1)
        ivec = vprefix.mean(dim=1)
        ivec = F.normalize(self.img_head(ivec), dim=-1)
        tvec = F.normalize(self.txt_head(tvec), dim=-1)
        sim = ivec @ tvec.t() / 0.07
        tar = torch.arange(B, device=sim.device)
        loss_itc = (F.cross_entropy(sim, tar) + F.cross_entropy(sim.t(), tar)) * 0.5

        loss = self.cfg.lambda_itc * loss_itc + self.cfg.lambda_itm * loss_itm + self.cfg.lambda_itg * loss_itg
        return {"loss": loss, "loss_itc": loss_itc, "loss_itm": loss_itm, "loss_itg": loss_itg}


    @torch.no_grad()
    def generate(self, images: torch.Tensor, sam_masks: Optional[torch.Tensor] = None,
                 context_vec: Optional[torch.Tensor] = None, prompt: str = "Describe the scene: "):
        tokens_I, EG, ES = self.visual_paths(images, sam_masks)
        EF = self.dual(EG, ES)
        Ktokens = self.ctx_gate(tokens_I, C=EF if context_vec is None else context_vec)
        Kq = self.vis2q(Ktokens)
        Qout = self._qformer_forward(Kq)
        vprefix = self.to_llm(Qout)

        toks = self.tokenizer([prompt] * images.size(0), return_tensors="pt", padding=True).to(images.device)
        emb = self.llm.get_input_embeddings()(toks.input_ids)
        emb = torch.cat([vprefix, emb], dim=1)
        pfx_mask = torch.ones(images.size(0), vprefix.size(1), device=emb.device, dtype=toks.attention_mask.dtype)
        attn = torch.cat([pfx_mask, toks.attention_mask], dim=1)

        out_ids = self.llm.generate(inputs_embeds=emb, attention_mask=attn, max_new_tokens=self.cfg.max_gen_len, do_sample=False)
        return self.tokenizer.batch_decode(out_ids, skip_special_tokens=True)

    @torch.no_grad()
    def encode_image(self, images: torch.Tensor, sam_masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        tokens_I, EG, ES = self.visual_paths(images, sam_masks)
        EF = self.dual(EG, ES)
        Ktokens = self.ctx_gate(tokens_I, C=EF)
        Kq = self.vis2q(Ktokens)
        Qout = self._qformer_forward(Kq).mean(dim=1)  # [B,d_q]
        v = self.to_llm(Qout).mean(dim=1)             # [B,d_L]
        return F.normalize(self.img_head(v), dim=-1)

    @torch.no_grad()
    def encode_text(self, texts: List[str], device) -> torch.Tensor:
        toks = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
        out = self.llm(**toks, output_hidden_states=True, use_cache=False)
        h = out.hidden_states[-1].mean(dim=1)
        return F.normalize(self.txt_head(h), dim=-1)


@torch.no_grad()
def recall_at_k(img_z: torch.Tensor, txt_z: torch.Tensor, k: int) -> float:
    sim = img_z @ txt_z.t()
    topk = sim.topk(k, dim=1).indices
    hits = torch.arange(sim.size(0), device=sim.device).unsqueeze(1)
    return (topk == hits).any(dim=1).float().mean().item()
