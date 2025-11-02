"""
Chronos-2 specific QA head for direct integration with LLaMA.

Simpler than the original QA head - designed specifically for Chronos-2 encoder output.
No channel fusion needed since we keep all Chronos embeddings.
"""

import os
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer

from .qa import LinearWithLoRA, _set_module  # Reuse LoRA implementation


class Chronos2QAHead(nn.Module):
    """
    Simplified QA head for Chronos-2 encoder.

    Directly projects Chronos-2 embeddings to LLaMA space without channel fusion.
    Keeps all temporal and channel information from Chronos-2.
    """

    def __init__(
        self,
        llama_model_name: str,
        chronos_dim: int = 768,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        use_lora: bool = True,
        log_mode: str = "info",
        device: str = "cuda",
    ):
        super().__init__()

        self.log_mode = log_mode
        self.chronos_dim = chronos_dim
        self.device = device

        # Load LLaMA tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(llama_model_name)
        special_tokens = {"additional_special_tokens": ["[SENSOR]"]}
        self.tokenizer.add_special_tokens(special_tokens)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load LLaMA model and move to device
        self.llama = AutoModelForCausalLM.from_pretrained(llama_model_name)
        self.llama.resize_token_embeddings(len(self.tokenizer))
        self.llama.to(device)

        # Get LLaMA hidden size
        hidden_size = self.llama.config.hidden_size

        # Simple projection: Chronos dim → LLaMA dim
        self.projector = nn.Sequential(
            nn.LayerNorm(chronos_dim),
            nn.Linear(chronos_dim, hidden_size),
        ).to(device)

        self._log(f"[Chronos2QAHead] Initialized: Chronos {chronos_dim} → LLaMA {hidden_size}", "info")
        self._log(f"[Chronos2QAHead] Device: {device}", "info")

        # Apply LoRA if requested
        if use_lora:
            self._enable_lora_adapters(lora_rank, lora_alpha, lora_dropout)
            self._log(f"[Chronos2QAHead] Enabled LoRA: rank={lora_rank}, alpha={lora_alpha}", "info")

    def _log(self, message: str, level: str = "info"):
        """Log message based on log_mode."""
        if level == "error":
            print(message)
        elif level == "warn" and self.log_mode != "silent":
            print(message)
        elif level == "info" and self.log_mode in {"info", "debug"}:
            print(message)
        elif level == "debug" and self.log_mode == "debug":
            print(message)

    def _enable_lora_adapters(self, rank: int, alpha: int, dropout: float):
        """Enable LoRA adapters on LLaMA."""
        self.llama.config.use_cache = False
        if hasattr(self.llama, "gradient_checkpointing_enable"):
            self.llama.gradient_checkpointing_enable()

        # Freeze all LLaMA parameters
        for param in self.llama.parameters():
            param.requires_grad = False

        # Replace target layers with LoRA
        target_suffixes = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
        replacements: List[Tuple[str, nn.Linear]] = []

        for name, module in self.llama.named_modules():
            if isinstance(module, nn.Linear) and name.endswith(target_suffixes):
                replacements.append((name, module))

        for name, base_linear in replacements:
            lora_linear = LinearWithLoRA(base_linear, rank, alpha, dropout)
            lora_linear.to(self.device)  # Move LoRA parameters to device
            _set_module(self.llama, name, lora_linear)

        self._log(f"[Chronos2QAHead] LoRA enabled on {len(replacements)} layers", "info")

    def prepare_text_batch(
        self,
        questions: List[str],
        answers: List[str],
        sensor_counts: torch.Tensor,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare text batch with sensor token placeholders.

        Args:
            questions: List of question strings
            answers: List of answer strings
            sensor_counts: (B,) tensor with number of sensor embeddings per sample
            device: Device to create tensors on

        Returns:
            Dict with input_ids, attention_mask, labels, sensor_positions
        """
        batch_size = len(questions)
        max_len = 0
        prepared = []

        for i, (q, a) in enumerate(zip(questions, answers)):
            # Build sequence: "Question: {q} [SENSOR]×N Answer: {a}"
            prefix = f"Question: {q} "
            prefix_ids = self.tokenizer(prefix, add_special_tokens=False).input_ids

            # Sensor tokens
            num_sensors = int(sensor_counts[i].item())
            sensor_token_id = self.tokenizer.convert_tokens_to_ids("[SENSOR]")
            sensor_ids = [sensor_token_id] * num_sensors

            # Answer prompt
            answer_prompt = " Answer:"
            answer_prompt_ids = self.tokenizer(answer_prompt, add_special_tokens=False).input_ids

            # Answer text
            answer_text = " " + a + self.tokenizer.eos_token
            answer_ids = self.tokenizer(answer_text, add_special_tokens=False).input_ids

            # Concatenate
            seq = prefix_ids + sensor_ids + answer_prompt_ids + answer_ids
            sensor_positions = list(range(len(prefix_ids), len(prefix_ids) + len(sensor_ids)))

            # Labels: only predict answer tokens
            labels = (
                [-100] * (len(prefix_ids) + len(sensor_ids) + len(answer_prompt_ids))
                + answer_ids
            )

            max_len = max(max_len, len(seq))
            prepared.append((seq, labels, sensor_positions))

        # Pad sequences
        pad_id = self.tokenizer.pad_token_id
        input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
        labels_tensor = torch.full((batch_size, max_len), -100, dtype=torch.long, device=device)
        sensor_positions_list = []

        for i, (seq, labels, sensor_pos) in enumerate(prepared):
            seq_len = len(seq)
            input_ids[i, :seq_len] = torch.tensor(seq, dtype=torch.long, device=device)
            attention_mask[i, :seq_len] = 1
            labels_tensor[i, :seq_len] = torch.tensor(labels, dtype=torch.long, device=device)
            sensor_positions_list.append(torch.tensor(sensor_pos, dtype=torch.long, device=device))

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels_tensor,
            "sensor_positions": sensor_positions_list,
        }

    def forward(
        self,
        chronos_embeddings: torch.Tensor,
        questions: List[str],
        answers: List[str],
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass through QA head.

        Args:
            chronos_embeddings: (B, seq_len, chronos_dim) embeddings from Chronos-2
            questions: List of question strings
            answers: List of answer strings

        Returns:
            loss: Scalar loss
            info: Dict with logits, labels, etc.
        """
        device = chronos_embeddings.device
        B, seq_len, _ = chronos_embeddings.shape

        # All samples have same number of sensor embeddings (seq_len)
        sensor_counts = torch.full((B,), seq_len, dtype=torch.long, device=device)

        # Prepare text batch
        tokenized = self.prepare_text_batch(questions, answers, sensor_counts, device)

        # Project Chronos embeddings to LLaMA space
        projected = self.projector(chronos_embeddings)  # (B, seq_len, llama_hidden)

        # Get base text embeddings
        base_embeds = self.llama.get_input_embeddings()(tokenized["input_ids"])
        if base_embeds.dtype != projected.dtype:
            base_embeds = base_embeds.to(projected.dtype)

        # Clone for injection
        embeds = base_embeds.clone()

        # Inject sensor embeddings at [SENSOR] token positions
        for i, positions in enumerate(tokenized["sensor_positions"]):
            num_positions = positions.numel()
            if num_positions == 0:
                continue

            # Use all available sensor embeddings
            sensor_embeds = projected[i]  # (seq_len, llama_hidden)

            # Ensure we have enough embeddings
            if sensor_embeds.size(0) < num_positions:
                # Pad if needed
                pad_len = num_positions - sensor_embeds.size(0)
                pad_tensor = torch.zeros((pad_len, sensor_embeds.size(-1)), device=device, dtype=sensor_embeds.dtype)
                sensor_embeds = torch.cat([sensor_embeds, pad_tensor], dim=0)
            elif sensor_embeds.size(0) > num_positions:
                # Truncate if needed
                sensor_embeds = sensor_embeds[:num_positions]

            # Inject
            embeds[i, positions, :] = sensor_embeds

        # Forward through LLaMA
        outputs = self.llama(
            inputs_embeds=embeds,
            attention_mask=tokenized["attention_mask"],
            labels=tokenized["labels"],
        )

        info = {
            "logits": outputs.logits,
            "labels": tokenized["labels"],
            "label_mask": tokenized["labels"] != -100,
        }

        return outputs.loss, info

    def generate(
        self,
        chronos_embeddings: torch.Tensor,
        questions: List[str],
        max_new_tokens: int = 32,
        temperature: float = 0.7,
        do_sample: bool = False,
    ) -> List[str]:
        """
        Generate answers autoregressively.

        Args:
            chronos_embeddings: (B, seq_len, chronos_dim) embeddings from Chronos-2
            questions: List of question strings
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to sample or use greedy

        Returns:
            List of generated answer strings
        """
        device = chronos_embeddings.device
        B, seq_len, _ = chronos_embeddings.shape

        # Prepare prompt (no answers)
        sensor_counts = torch.full((B,), seq_len, dtype=torch.long, device=device)
        dummy_answers = [""] * B
        tokenized = self.prepare_text_batch(questions, dummy_answers, sensor_counts, device)

        # Project embeddings
        projected = self.projector(chronos_embeddings)

        # Get base embeddings and inject sensor embeddings
        base_embeds = self.llama.get_input_embeddings()(tokenized["input_ids"])
        if base_embeds.dtype != projected.dtype:
            base_embeds = base_embeds.to(projected.dtype)
        embeds = base_embeds.clone()

        for i, positions in enumerate(tokenized["sensor_positions"]):
            num_positions = positions.numel()
            if num_positions == 0:
                continue

            sensor_embeds = projected[i]
            if sensor_embeds.size(0) < num_positions:
                pad_len = num_positions - sensor_embeds.size(0)
                pad_tensor = torch.zeros((pad_len, sensor_embeds.size(-1)), device=device, dtype=sensor_embeds.dtype)
                sensor_embeds = torch.cat([sensor_embeds, pad_tensor], dim=0)
            elif sensor_embeds.size(0) > num_positions:
                sensor_embeds = sensor_embeds[:num_positions]

            embeds[i, positions, :] = sensor_embeds

        # Generate
        generated_ids = self.llama.generate(
            inputs_embeds=embeds,
            attention_mask=tokenized["attention_mask"],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # Decode (skip prompt)
        prompt_length = tokenized["input_ids"].shape[1]
        generated_texts = []
        for ids in generated_ids:
            new_tokens = ids[prompt_length:]
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            generated_texts.append(text.strip())

        return generated_texts

    def save_checkpoint(self, out_dir: str, epoch: int):
        """Save checkpoint."""
        os.makedirs(out_dir, exist_ok=True)
        head_path = os.path.join(out_dir, f"chronos2_qa_head_e{epoch}.pt")
        torch.save(self.state_dict(), head_path)
        tokenizer_dir = os.path.join(out_dir, f"tokenizer_e{epoch}")
        self.tokenizer.save_pretrained(tokenizer_dir)
        self._log(f"[SAVE] QA head → {head_path}, tokenizer → {tokenizer_dir}", "info")
