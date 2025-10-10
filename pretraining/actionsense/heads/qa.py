import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer


class LinearWithLoRA(nn.Module):
    """Wrap an nn.Linear with a LoRA adapter (rank-r low-rank residual)."""

    def __init__(
        self,
        base_linear: nn.Linear,
        rank: int,
        alpha: int,
        dropout: float,
    ) -> None:
        super().__init__()

        if rank <= 0:
            raise ValueError("LoRA rank must be > 0")

        self.base = base_linear
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        in_features = self.base.in_features
        out_features = self.base.out_features

        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.lora_A, mean=0.0, std=0.01)
        nn.init.zeros_(self.lora_B)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        base_out = self.base(input)
        lora_intermediate = F.linear(self.dropout(input), self.lora_A)
        lora_out = F.linear(lora_intermediate, self.lora_B) * self.scaling
        return base_out + lora_out


def _set_module(root: nn.Module, name: str, module: nn.Module) -> None:
    parts = name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], module)


@dataclass
class TokenizedBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    sensor_positions: List[torch.Tensor]


class SensorQALLMHead(nn.Module):
    """Fuses channel tokens per patch, projects to LLM space, and computes QA loss."""

    def __init__(
        self,
        llama_model_name: str,
        feature_dim: int,
        attn_heads: int = 4,
        attn_dropout: float = 0.1,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        use_lora: bool = True,
        log_mode: str = "info",
    ) -> None:
        super().__init__()

        self.log_mode = log_mode

        self.tokenizer = AutoTokenizer.from_pretrained(llama_model_name)
        special_tokens = {"additional_special_tokens": ["[SENSOR]"]}
        self.tokenizer.add_special_tokens(special_tokens)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.llama = AutoModelForCausalLM.from_pretrained(llama_model_name)
        self.llama.resize_token_embeddings(len(self.tokenizer))

        self.channel_query = nn.Parameter(torch.randn(1, 1, feature_dim))
        self.channel_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=attn_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.channel_norm = nn.LayerNorm(feature_dim)

        hidden_size = self.llama.config.hidden_size
        self.projector = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_size),
        )
        self._debug_logged = False
        self._token_debug_logged = False
        self._forward_calls = 0

        if use_lora:
            self._enable_lora_adapters(lora_rank, lora_alpha, lora_dropout)

    def _log(self, message: str, level: str = "info") -> None:
        mode = getattr(self, "log_mode", "info")
        if level == "error":
            print(message)
            return
        if level == "warn":
            if mode != "silent":
                print(message)
            return
        if level == "info":
            if mode in {"info", "debug"}:
                print(message)
            return
        if level == "debug":
            if mode == "debug":
                print(message)

    def fuse_channels(self, tokens: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """Reduce channel dimension with attention pooling (B,P,D,F) -> (B,P,F)."""
        B, P, D, F = tokens.shape
        tokens_flat = tokens.view(B * P, D, F)
        query = self.channel_query.expand(tokens_flat.size(0), -1, -1)
        pooled, _ = self.channel_attn(query, tokens_flat, tokens_flat)
        fused = pooled.view(B, P, F)

        fused = self.channel_norm(fused)
        if pad_mask is not None:
            fused = fused * pad_mask.unsqueeze(-1).to(fused.dtype)
        return fused

    def _enable_lora_adapters(self, rank: int, alpha: int, dropout: float) -> None:
        self.llama.config.use_cache = False
        if hasattr(self.llama, "gradient_checkpointing_enable"):
            self.llama.gradient_checkpointing_enable()

        for param in self.llama.parameters():
            param.requires_grad = False

        target_suffixes = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
        replacements: List[Tuple[str, nn.Linear]] = []
        for name, module in self.llama.named_modules():
            if isinstance(module, nn.Linear) and name.endswith(target_suffixes):
                replacements.append((name, module))

        for name, module in replacements:
            lora_module = LinearWithLoRA(module, rank=rank, alpha=alpha, dropout=dropout)
            _set_module(self.llama, name, lora_module)

        if replacements:
            self._log(
                f"[INFO] Enabled LoRA adapters (rank={rank}, alpha={alpha}, dropout={dropout}) on {len(replacements)} linear layers",
                level="info",
            )
        else:
            self._log("[WARN] No target modules found for LoRA; base model remains frozen", level="warn")

    def prepare_text_batch(
        self,
        questions: List[str],
        answers: List[str],
        sensor_counts: torch.Tensor,
        device: torch.device,
    ) -> TokenizedBatch:
        batch_size = len(questions)
        prepared: List[Tuple[List[int], List[int], List[int]]] = []
        max_len = 0
        sensor_token_id = self.tokenizer.convert_tokens_to_ids("[SENSOR]")

        for q, a, count in zip(questions, answers, sensor_counts.tolist()):
            prefix = (
                f"QUESTION: {q}\n"
                "INSTRUCTION: Answer the question using the available context.\n"
                "SENSOR_CONTEXT:"
            )
            prefix_ids = self.tokenizer(prefix, add_special_tokens=False).input_ids
            num_sensor_tokens = max(int(count), 1)
            sensor_ids = [sensor_token_id] * num_sensor_tokens
            answer_prompt_ids = self.tokenizer("\nANSWER:", add_special_tokens=False).input_ids
            answer_text = " " + a + self.tokenizer.eos_token
            answer_ids = self.tokenizer(answer_text, add_special_tokens=False).input_ids

            seq = prefix_ids + sensor_ids + answer_prompt_ids + answer_ids
            sensor_positions = list(range(len(prefix_ids), len(prefix_ids) + len(sensor_ids)))
            labels = (
                [-100] * (len(prefix_ids) + len(sensor_ids) + len(answer_prompt_ids))
                + answer_ids
            )
            max_len = max(max_len, len(seq))
            prepared.append((seq, labels, sensor_positions))

        pad_id = self.tokenizer.pad_token_id
        input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
        labels_tensor = torch.full((batch_size, max_len), -100, dtype=torch.long, device=device)
        sensor_positions: List[torch.Tensor] = []

        for i, (seq, labels, sensor_pos) in enumerate(prepared):
            seq_len = len(seq)
            input_ids[i, :seq_len] = torch.tensor(seq, dtype=torch.long, device=device)
            attention_mask[i, :seq_len] = 1
            labels_tensor[i, :seq_len] = torch.tensor(labels, dtype=torch.long, device=device)
            sensor_positions.append(torch.tensor(sensor_pos, dtype=torch.long, device=device))

        if not self._token_debug_logged and self.log_mode == "debug":
            for i, (seq, _labels, sensor_pos) in enumerate(prepared):
                self._log(
                    f"[DEBUG] Tokenization sample {i}: seq_len={len(seq)}, sensor_token_count={len(sensor_pos)}",
                    level="debug",
                )
            self._log(
                f"[DEBUG] Tokenized batch tensors -> input_ids shape={input_ids.shape}, attention_mask shape={attention_mask.shape}, labels shape={labels_tensor.shape}",
                level="debug",
            )
            self._token_debug_logged = True

        return TokenizedBatch(input_ids, attention_mask, labels_tensor, sensor_positions)

    def forward(
        self,
        tokens: torch.Tensor,
        pad_mask: torch.Tensor,
        questions: List[str],
        answers: List[str],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        device = tokens.device
        fused = self.fuse_channels(tokens, pad_mask)  # (B,P,F)
        sensor_counts = pad_mask.sum(dim=1) if pad_mask is not None else torch.full(
            (tokens.size(0),), tokens.size(1), device=device, dtype=torch.long
        )
        sensor_counts = sensor_counts.to(torch.long)

        tokenized = self.prepare_text_batch(questions, answers, sensor_counts, device)

        projected = self.projector(fused)  # (B,P,H)
        base_embeds = self.llama.get_input_embeddings()(tokenized.input_ids)
        if base_embeds.dtype != projected.dtype:
            base_embeds = base_embeds.to(projected.dtype)
        embeds = base_embeds.clone()
        if not self._debug_logged and self.log_mode == "debug":
            self._log(
                f"[DEBUG] Fused shape={fused.shape} dtype={fused.dtype} projected shape={projected.shape} dtype={projected.dtype}",
                level="debug",
            )
            self._log(
                f"[DEBUG] Tokenized ids shape={tokenized.input_ids.shape} embeds shape={embeds.shape} dtype={embeds.dtype}",
                level="debug",
            )
            self._debug_logged = True

        for i, positions in enumerate(tokenized.sensor_positions):
            count = positions.numel()
            if count == 0:
                continue
            valid_count = int(sensor_counts[i].item()) if sensor_counts is not None else count
            valid_count = max(1, min(valid_count, count))
            patch_embeds = projected[i, :valid_count, :]
            if patch_embeds.dtype != embeds.dtype:
                patch_embeds = patch_embeds.to(embeds.dtype)
            if patch_embeds.size(0) < count:
                pad_len = count - patch_embeds.size(0)
                pad_tensor = torch.zeros((pad_len, patch_embeds.size(-1)), device=device, dtype=patch_embeds.dtype)
                patch_embeds = torch.cat([patch_embeds, pad_tensor], dim=0)
            elif patch_embeds.size(0) > count:
                patch_embeds = patch_embeds[:count]
            embeds[i, positions, :] = patch_embeds

        outputs = self.llama(
            inputs_embeds=embeds,
            attention_mask=tokenized.attention_mask,
            labels=tokenized.labels,
        )
        self._forward_calls += 1

        if self._forward_calls % 10 == 0 and self.log_mode == "debug":
            with torch.no_grad():
                pred_ids = outputs.logits.argmax(dim=-1)
                label_mask = tokenized.labels != -100
                for i in range(min(len(questions), 2)):
                    mask = label_mask[i]
                    if mask.any():
                        generated_ids = pred_ids[i][mask].detach().cpu()
                        generated_text = self.tokenizer.decode(generated_ids.tolist(), skip_special_tokens=True)
                        gt_ids = tokenized.labels[i][mask].detach().cpu()
                        gt_text = self.tokenizer.decode(gt_ids.tolist(), skip_special_tokens=True)
                        self._log(f"[INFO] Batch gen[{i}] -> {generated_text}", level="info")
                        self._log(f"[INFO] Batch gt [{i}] -> {gt_text}", level="info")
        info = {
            "logits": outputs.logits,
            "labels": tokenized.labels,
            "label_mask": tokenized.labels != -100,
            "fused_patches": fused.detach(),
        }
        return outputs.loss, info

    def save_checkpoint(self, out_dir: str, epoch: int) -> None:
        os.makedirs(out_dir, exist_ok=True)
        head_path = os.path.join(out_dir, f"qa_head_e{epoch}.pt")
        torch.save(self.state_dict(), head_path)
        tokenizer_dir = os.path.join(out_dir, f"tokenizer_e{epoch}")
        self.tokenizer.save_pretrained(tokenizer_dir)
        self._log(f"[INFO] qa_head -> {head_path}\n[SAVE] tokenizer -> {tokenizer_dir}", level="info")
