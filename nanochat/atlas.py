"""
Atlas wrapper for nanochat training infrastructure.

Uses AtlasLMM (pure memory, no attention) from atlas-pytorch (external, verified)
with nanochat's tokenizer, dataloader, optimizer, and evaluation infrastructure.

This replaces the local Atlas implementation with the verified atlas-pytorch library.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from atlas_pytorch import AtlasLMM, MemoryMLP

from nanochat.common import get_dist_info, print0, COMPUTE_DTYPE
from nanochat.optim import MuonAdamW, DistMuonAdamW


@dataclass
class AtlasConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 24
    n_head: int = 16
    n_embd: int = 1536
    dim_head: int = 64
    neural_mem_depth: int = 2
    omega_window: int = 16
    poly_degree: int = 3
    poly_mode: str = "elementwise"
    ff_mult: int = 4
    num_persist_mem_tokens: int = 0
    use_accelerated_scan: bool = False


class Atlas(nn.Module):
    """Wraps atlas-pytorch AtlasLMM for nanochat's training loop.

    nanochat expects:
      - model(x, y) -> scalar loss
      - model(x, y, loss_reduction='none') -> (B, T) per-token loss
      - model.config, model.estimate_flops(), model.num_scaling_params()
      - model.setup_optimizer(...), model.init_weights()
      - model.generate(tokens, max_tokens, temperature, top_k, seed)
    """

    def __init__(self, config: AtlasConfig):
        super().__init__()
        self.config = config

        neural_memory_model = MemoryMLP(
            dim=config.dim_head,
            depth=config.neural_mem_depth,
        )

        self.model = AtlasLMM(
            num_tokens=config.vocab_size,
            dim=config.n_embd,
            depth=config.n_layer,
            num_persist_mem_tokens=config.num_persist_mem_tokens,
            neural_memory_model=neural_memory_model,
            ff_mult=config.ff_mult,
            omega_window=config.omega_window,
            poly_degree=config.poly_degree,
            poly_mode='off' if config.poly_degree == 0 else config.poly_mode,
            use_muon_optimizer=False,
            use_omega_gate=False,
            neural_memory_kwargs=dict(
                dim_head=config.dim_head,
                heads=config.n_head,
                qk_rmsnorm=True,
                momentum=True,
                momentum_order=1,
                default_step_transform_max_lr=1e-1,
                use_accelerated_scan=config.use_accelerated_scan,
            ),
        )

    @torch.no_grad()
    def init_weights(self):
        """Re-initialize weights following nanochat conventions."""
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5

        if hasattr(self.model, 'token_emb'):
            nn.init.normal_(self.model.token_emb.weight, mean=0.0, std=0.8)

        if hasattr(self.model, 'to_logits'):
            for m in self.model.to_logits.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0.0, std=0.001)

        self.to(dtype=COMPUTE_DTYPE)

    def get_device(self):
        return next(self.parameters()).device

    def forward(self, idx, targets=None, loss_reduction='mean'):
        B, T = idx.size()

        logits = self.model(idx)  # (B, T, V)

        if targets is not None:
            softcap = 15
            logits_f = logits.float()
            logits_f = softcap * torch.tanh(logits_f / softcap)

            loss = F.cross_entropy(
                logits_f.view(-1, logits_f.size(-1)),
                targets.view(-1),
                ignore_index=-1,
                reduction=loss_reduction,
            )
            if loss_reduction == 'none':
                loss = loss.view(B, T)
            return loss
        return logits

    def estimate_flops(self):
        cfg = self.config
        nparams = sum(p.numel() for p in self.parameters())
        nembed = cfg.vocab_size * cfg.n_embd
        base_flops = 6 * (nparams - nembed)

        # Memory module FLOPs per token per layer
        H = cfg.n_head
        D = cfg.dim_head
        mem_flops_per_layer = H * D * D * 5
        total_mem_flops = mem_flops_per_layer * cfg.n_layer

        return base_flops + total_mem_flops

    def num_scaling_params(self):
        embed_params = 0
        lm_head_params = 0
        if hasattr(self.model, 'token_emb'):
            embed_params = sum(p.numel() for p in self.model.token_emb.parameters())
        if hasattr(self.model, 'to_logits'):
            lm_head_params = sum(p.numel() for p in self.model.to_logits.parameters())

        total = sum(p.numel() for p in self.parameters())
        transformer_matrices = total - embed_params - lm_head_params

        return {
            'wte': embed_params,
            'lm_head': lm_head_params,
            'transformer_matrices': transformer_matrices,
            'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.008, embedding_lr=0.3, matrix_lr=0.02,
                        weight_decay=0.0, scalar_lr=0.5):
        ddp, rank, local_rank, world_size = get_dist_info()
        model_dim = self.config.n_embd

        embedding_params = []
        lm_head_params = []
        matrix_params = []
        small_params = []

        if hasattr(self.model, 'token_emb'):
            embedding_params.extend(list(self.model.token_emb.parameters()))
        if hasattr(self.model, 'to_logits'):
            lm_head_params.extend(list(self.model.to_logits.parameters()))

        tiny_params = []
        memory_model_params = []  # internal memory weights — no standard gradients
        embed_ids = {id(p) for p in embedding_params}
        head_ids = {id(p) for p in lm_head_params}
        for name, p in self.model.named_parameters():
            if id(p) in embed_ids or id(p) in head_ids:
                continue
            # memory_model internals + omega_gate are updated by the memory mechanism, not backprop
            if 'memory_model.' in name or '_omega_gate_' in name:
                memory_model_params.append(p)
                continue
            if p.ndim == 2 and min(p.shape) >= 64:
                matrix_params.append(p)
            elif p.shape[0] < max(world_size, 4) or (world_size > 1 and p.shape[0] % world_size != 0):
                tiny_params.append(p)  # too small or indivisible for reduce_scatter
            else:
                small_params.append(p)

        all_count = len(embedding_params) + len(lm_head_params) + len(matrix_params) + len(small_params) + len(tiny_params) + len(memory_model_params)
        total_count = len(list(self.parameters()))
        assert all_count == total_count, f"Parameter grouping mismatch: {all_count} vs {total_count}"
        if memory_model_params:
            print0(f"Atlas: excluding {len(memory_model_params)} internal memory params from optimizer (updated by memory mechanism)")

        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Atlas: Scaling LR ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")
        print0(f"Atlas params: embed={len(embedding_params)}, lm_head={len(lm_head_params)}, matrix={len(matrix_params)}, small={len(small_params)}, tiny={len(tiny_params)}")

        param_groups = [
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale,
                 betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale,
                 betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            dict(kind='adamw', params=small_params, lr=scalar_lr * 0.1,
                 betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
        ]

        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=weight_decay,
            ))

        Factory = DistMuonAdamW if ddp else MuonAdamW
        optimizer = Factory(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        assert isinstance(tokens, list)
        device = self.get_device()

        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)

        ids = torch.tensor([tokens], dtype=torch.long, device=device)

        for _ in range(max_tokens):
            logits = self.model(ids)  # (1, T, V)
            logits = logits[:, -1, :]

            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)

            token = next_id.item()
            yield token
            ids = torch.cat([ids, next_id], dim=1)
