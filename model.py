import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import CLIPModel, AutoModelForCausalLM, AutoTokenizer


@dataclass
class MDSEConfig:
    clip_name: str = "openai/clip-vit-large-patch14"  # ViT-L/14
    image_size: int = 224

    # Q-Former (BLIP-2 style adaptor)
    q_layers: int = 12
    q_heads: int = 12
    q_dim: int = 768          # d_q
    q_mlp: int = 3072
    num_queries: int = 32     # N queries

    # LLM
    llm_name: str = "mistralai/Mistral-7B-Instruct-v0.2"
    # proj to LLM hidden size inferred from model
    max_gen_len: int = 32

    # Loss weights (λ1, λ2, λ3) in Eq. (5)
    lambda_itc: float = 1.0   # contrastive
    lambda_itm: float = 0.5   # matching
    lambda_itg: float = 0.1   # generation

    # LoRA (Section 3.4) — set enable_lora=True and attach with PEFT outside if desired
    enable_lora: bool = False
    lora_r_qformer: int = 8
    lora_alpha_qformer: int = 16
    lora_r_llm_proj: int = 4
    lora_alpha_llm_proj: int = 8
    lora_dropout: float = 0.05


# ----------------------------
# 3.1 Feature Enhancement FE(I) -> Î  (multi-exposure fusion, tone map, local contrast, WB, saturation)
# (Implementation is a compact, differentiable approximation of the described pipeline.)
# ----------------------------
class FeatureEnhancement(nn.Module):
    def __init__(self):
        super().__init__()
        # Learnable scalars (kept small) for light, contrast, saturation
        self.gamma = nn.Parameter(torch.tensor(0.9))   # spatial tone mapping (γ<1 boosts low light)
        self.sharp = nn.Parameter(torch.tensor(0.12))  # local contrast (unsharp)
        self.sat   = nn.Parameter(torch.tensor(1.05))  # saturation scaling
        self.register_buffer("eps", torch.tensor(1e-6))

    def multi_exposure_fusion(self, x: torch.Tensor) -> torch.Tensor:
        # Simulate 3 exposures via gamma; fuse with well-exposedness & contrast weights.
        g = [0.7, 1.0, 1.4]
        expos = [torch.clamp(x ** gk, 0, 1) for gk in g]
        # well-exposedness
        mu = 0.5
        ex_w = [torch.exp(-0.5 * ((e - mu) ** 2) / (0.2 ** 2)).mean(dim=1, keepdim=True) for e in expos]
        # local contrast (Laplacian magnitude over intensity)
        gray = [e.mean(dim=1, keepdim=True) for e in expos]
        lap  = [F.laplacian(gv) if hasattr(F, "laplacian") else F.conv2d(gv, weight=torch.tensor(
               [[[[0,-1,0],[-1,4,-1],[0,-1,0]]]], device=gv.device, dtype=gv.dtype), padding=1) for gv in gray]
        ct_w = [l.abs() for l in lap]
        W = [ex + ct for ex, ct in zip(ex_w, ct_w)]
        W = [w / (w.mean(dim=(2,3), keepdim=True) + self.eps) for w in W]
        num = sum([w * e for w, e in zip(W, expos)])
        den = sum(W) + self.eps
        return torch.clamp(num / den, 0, 1)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        x = img.clamp(0, 1)
        # (a) multi-exposure fusion
        x = self.multi_exposure_fusion(x)
        # (b) spatial tone mapping (γ)
        x = torch.pow(x + self.eps, self.gamma)
        # (c) local contrast via unsharp mask
        blur = F.avg_pool2d(x, 3, 1, 1)
        x = torch.clamp((1 + self.sharp) * x - self.sharp * blur, 0, 1)
        # (d) white balance (gray-world + white-point)
        mean = x.mean(dim=(2,3), keepdim=True) + self.eps
        gain = mean.mean(dim=1, keepdim=True) / mean
        x = torch.clamp(x * gain, 0, 1)
        # (e) saturation scaling (simple HSV approximation)
        maxc, _ = x.max(dim=1, keepdim=True)
        minc, _ = x.min(dim=1, keepdim=True)
        s = (maxc - minc) / (maxc + self.eps)
        s = torch.clamp(s * self.sat, 0, 1)
        x = (x - minc) * (s / (s + self.eps)) + minc
        return x.clamp(0, 1)


# ----------------------------
# SAM-based segmentation (offline) → masks list; here we only *consume* masks.
# We build ES by masking Î with each region and encoding via CLIP; average across regions.
# ----------------------------
def encode_regions_with_clip(clip_vision, pixel_values: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """
    pixel_values: [B,3,H,W], masks: [B,K,H,W] in {0,1}
    returns ES pooled embedding: [B, Dv]
    """
    B, K = masks.shape[:2]
    if K == 0:
        # fall back to zeros; caller can handle
        Dv = clip_vision.config.hidden_size
        return torch.zeros(B, Dv, device=pixel_values.device, dtype=pixel_values.dtype)

    # Build masked images (sum over K, average embeddings)
    pv = pixel_values.unsqueeze(1) * masks.unsqueeze(2)  # [B,K,3,H,W]
    pv = pv.reshape(B * K, 3, pixel_values.size(2), pixel_values.size(3))
    out = clip_vision(pixel_values=pv)
    pooled = out.pooler_output  # [B*K, Dv]
    pooled = pooled.reshape(B, K, -1).mean(dim=1)        # ES: [B, Dv]
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
        self.f_ctx = nn.Linear(d_ctx, d_vis)      # f_ctx(C) -> C'
        self.g_proj = nn.Linear(d_vis + d_vis, d_vis)  # W_g on [E; C']

    def forward(self, E_tokens: torch.Tensor, C: Optional[torch.Tensor]) -> torch.Tensor:
        """
        E_tokens: [B, T, Dv] (global tokens from Î plus region tokens if desired)
        C: [B, d_ctx] or None
        returns K' tokens same shape
        """
        if C is None:
            return E_tokens
        Cprime = self.f_ctx(C)                     # [B, Dv]
        Cprime = Cprime.unsqueeze(1).expand_as(E_tokens)
        gate = torch.sigmoid(self.g_proj(torch.cat([E_tokens, Cprime], dim=-1)))
        return E_tokens + gate


# --- Minimal Q-Former (BLIP-2 style) with cross-attn to context-gated tokens ---
class QFormerBlock(nn.Module):
    def __init__(self, d: int, heads: int, mlp: int):
        super().__init__()
        self.ca = nn.MultiheadAttention(d, heads, batch_first=True)   # cross-attn Q <- K'V'
        self.sa = nn.MultiheadAttention(d, heads, batch_first=True)   # self-attn on queries
        self.ff = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, mlp), nn.GELU(), nn.Linear(mlp, d))
        self.ln1 = nn.LayerNorm(d); self.ln2 = nn.LayerNorm(d); self.ln3 = nn.LayerNorm(d)

    def forward(self, q: torch.Tensor, kvtoks: torch.Tensor):
        q = q + self.ca(self.ln1(q), self.ln1(kvtoks), self.ln1(kvtoks), need_weights=False)[0]
        q = q + self.sa(self.ln2(q), self.ln2(q), self.ln2(q), need_weights=False)[0]
        q = q + self.ff(q)
        return q


class QFormer(nn.Module):
    def __init__(self, num_queries: int, d_q: int, layers: int, heads: int, mlp: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(num_queries, d_q) / math.sqrt(d_q))
        self.blocks = nn.ModuleList([QFormerBlock(d_q, heads, mlp) for _ in range(layers)])
        self.ln_out = nn.LayerNorm(d_q)

    def forward(self, kvtoks: torch.Tensor):
        B = kvtoks.size(0)
        q = self.query.unsqueeze(0).expand(B, -1, -1)  # [B, N, d_q]
        for blk in self.blocks:
            q = blk(q, kvtoks)
        return self.ln_out(q)  # [B, N, d_q]


# ----------------------------
# MDSE main model
# ----------------------------
class MDSE(nn.Module):
    def __init__(self, cfg: MDSEConfig):
        super().__init__()
        self.cfg = cfg

        # FE
        self.fe = FeatureEnhancement()

        # Vision encoder (CLIP ViT-L/14)
        self.clip = CLIPModel.from_pretrained(cfg.clip_name)
        self.vis_dim = self.clip.vision_model.config.hidden_size  # e.g., 1024

        self.dual = DualPathway(self.vis_dim)
        self.ctx_gate = ContextGate(self.vis_dim, d_ctx=self.vis_dim)

        self.vis2q = nn.Linear(self.vis_dim, cfg.q_dim)
        self.qformer = QFormer(cfg.num_queries, cfg.q_dim, cfg.q_layers, cfg.q_heads, cfg.q_mlp)

        self.llm = AutoModelForCausalLM.from_pretrained(cfg.llm_name)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.llm_name)
        d_L = self.llm.config.hidden_size
        self.to_llm = nn.Linear(cfg.q_dim, d_L)


        self.itm_head = nn.Linear(cfg.q_dim, 2)

        self.txt_head = nn.Linear(d_L, d_L, bias=False)
        self.img_head = nn.Linear(d_L, d_L, bias=False)

    @torch.no_grad()
    def _clip_tokens_and_pool(self, pixel_values: torch.Tensor):
        out = self.clip.vision_model(pixel_values=pixel_values)
        tokens = out.last_hidden_state           # [B, T, Dv]
        pooled = out.pooler_output               # [B, Dv]
        return tokens, pooled

    def visual_paths(self, images: torch.Tensor, sam_masks: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        Ihat = self.fe(images)                                    # Î = FE(I)
        tokens, EG = self._clip_tokens_and_pool(Ihat)             # EG = VE(Î)
        if sam_masks is not None:
            ES = encode_regions_with_clip(self.clip.vision_model, Ihat, sam_masks)  # VE(M ⊙ Î)
        else:
            ES = torch.zeros_like(EG)
        return tokens, EG, ES

    # ----- Forward for captioning (teacher-forced) -----
    def forward_caption(self, images: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                        sam_masks: Optional[torch.Tensor] = None, context_vec: Optional[torch.Tensor] = None):
        tokens_I, EG, ES = self.visual_paths(images, sam_masks)
        EF = self.dual(EG, ES)                                  
        Ktokens = self.ctx_gate(tokens_I, C=EF if context_vec is None else context_vec)  # K' tokens
        Kq = self.vis2q(Ktokens)                                  # to Q-Former dim
        Qout = self.qformer(Kq)                                   # [B,N,d_q]
        vprefix = self.to_llm(Qout)                               # [B,N,d_L]

       
        B = images.size(0)
        emb = self.llm.get_input_embeddings()(input_ids)         # [B,L,d_L]
        emb = torch.cat([vprefix, emb], dim=1)                    # prepend visual prefix
        pfx_mask = torch.ones(B, vprefix.size(1), device=emb.device, dtype=attention_mask.dtype)
        attn = torch.cat([pfx_mask, attention_mask], dim=1)

        out = self.llm(inputs_embeds=emb, attention_mask=attn)
        logits = out.logits                                      


        L = input_ids.size(1)

        pred = logits[:, -L-1:-1, :]
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        loss_itg = F.cross_entropy(pred.reshape(-1, pred.size(-1)), labels.reshape(-1), ignore_index=-100)


        qavg = Qout.mean(dim=1)                                  
        itm_logits = self.itm_head(qavg)                          
 
        targets = torch.ones(B, dtype=torch.long, device=images.device)
        loss_itm = F.cross_entropy(itm_logits, targets)

        # (iii) ITC contrastive (encode image & text)
        with torch.no_grad():
            text_out = self.llm(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            tvec = text_out.hidden_states[-1].mean(dim=1)         # [B,d_L]
        ivec = vprefix.mean(dim=1)                                # [B,d_L]
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
        Qout = self.qformer(Kq)
        vprefix = self.to_llm(Qout)                               # [B,N,d_L]

        # Build prompt tokens then feed with visual prefix
        toks = self.tokenizer([prompt] * images.size(0), return_tensors="pt", padding=True).to(images.device)
        emb = self.llm.get_input_embeddings()(toks.input_ids)
        emb = torch.cat([vprefix, emb], dim=1)
        pfx_mask = torch.ones(images.size(0), vprefix.size(1), device=emb.device, dtype=toks.attention_mask.dtype)
        attn = torch.cat([pfx_mask, toks.attention_mask], dim=1)

        out_ids = self.llm.generate(inputs_embeds=emb, attention_mask=attn,
                                    max_new_tokens=self.cfg.max_gen_len, do_sample=False)
        return self.tokenizer.batch_decode(out_ids, skip_special_tokens=True)

   
    @torch.no_grad()
    def encode_image(self, images: torch.Tensor, sam_masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        tokens_I, EG, ES = self.visual_paths(images, sam_masks)
        EF = self.dual(EG, ES)
        Ktokens = self.ctx_gate(tokens_I, C=EF)
        Qout = self.qformer(self.vis2q(Ktokens)).mean(dim=1)
        v = self.to_llm(Qout).mean(dim=1)  # [B,d_L]
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
