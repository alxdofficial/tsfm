"""
MOMENT-specific QA head for question answering tasks.

Architecture:
1. Per-patch cross-channel attention: Fuse 18 channels into 1 token per patch
2. Projection: Map MOMENT embeddings to LLaMA space
3. LLaMA integration: Inject sensor embeddings for QA

Key difference from CLS head: Output is a SEQUENCE (64 tokens), not a single pooled token.
"""

import os
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

from .qa import LinearWithLoRA, _set_module  # Reuse LoRA implementation


class MOMENTQAHead(nn.Module):
    """
    QA head for MOMENT encoder with LLaMA integration.

    Pipeline:
        Input: (B, D=18, P=64, F) MOMENT embeddings
        → Step 1: Per-patch cross-channel attention → (B, P=64, F)
        → Step 2: Project to LLaMA space → (B, P=64, llama_dim)
        → Step 3: Inject into LLaMA for QA
    """

    def __init__(
        self,
        moment_dim: int,
        llama_model_name: str = "meta-llama/Llama-3.2-1B-Instruct",
        nhead: int = 8,
        dropout: float = 0.1,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        use_lora: bool = True,
        log_mode: str = "info",
        device: str = "cuda",
    ):
        """
        Args:
            moment_dim: MOMENT output dimension (512 for small, 768 for base, 1024 for large)
            llama_model_name: HuggingFace model name for LLaMA
            nhead: Number of attention heads for channel fusion
            dropout: Dropout probability
            lora_rank: LoRA rank
            lora_alpha: LoRA alpha scaling
            lora_dropout: LoRA dropout
            use_lora: Whether to use LoRA (recommended for memory efficiency)
            log_mode: Logging verbosity
            device: Device to load models on
        """
        super().__init__()

        self.moment_dim = moment_dim
        self.log_mode = log_mode
        self.device = device

        # Step 1: Per-patch cross-channel attention (from MOMENTCLSHead)
        # Learnable query to pool D=18 channels into 1 token per patch
        self.channel_query = nn.Parameter(torch.randn(1, 1, moment_dim))
        self.channel_attn = nn.MultiheadAttention(
            embed_dim=moment_dim,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.channel_norm = nn.LayerNorm(moment_dim)
        self.query_norm = nn.LayerNorm(moment_dim)

        self._log(f"[MOMENTQAHead] Channel fusion: D=18 → 1 per patch (nhead={nhead})", "info")

        # Load LLaMA tokenizer
        self._log(f"[MOMENTQAHead] Loading LLaMA tokenizer: {llama_model_name}", "info")
        self.tokenizer = AutoTokenizer.from_pretrained(llama_model_name)
        special_tokens = {"additional_special_tokens": ["[SENSOR]"]}
        self.tokenizer.add_special_tokens(special_tokens)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load LLaMA model
        self._log(f"[MOMENTQAHead] Loading LLaMA model: {llama_model_name}", "info")
        self.llama = AutoModelForCausalLM.from_pretrained(llama_model_name)
        self.llama.resize_token_embeddings(len(self.tokenizer))
        self.llama.to(device)

        # Get LLaMA hidden size
        llama_hidden_size = self.llama.config.hidden_size
        self._log(f"[MOMENTQAHead] LLaMA hidden size: {llama_hidden_size}", "info")

        # Step 2: Projection layer (MOMENT dim → LLaMA dim)
        self.projector = nn.Sequential(
            nn.LayerNorm(moment_dim),
            nn.Linear(moment_dim, llama_hidden_size),
        ).to(device)

        self._log(f"[MOMENTQAHead] Projector: {moment_dim} → {llama_hidden_size}", "info")

        # Apply LoRA if requested
        if use_lora:
            self._enable_lora_adapters(lora_rank, lora_alpha, lora_dropout)
            self._log(f"[MOMENTQAHead] LoRA enabled: rank={lora_rank}, alpha={lora_alpha}", "info")

        self._log(f"[MOMENTQAHead] Initialized successfully", "info")
        self._log(f"  Input:  (B, D=18, P=64, F={moment_dim})", "info")
        self._log(f"  Output: 64 sensor tokens in LLaMA space", "info")

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
        """Enable LoRA adapters on LLaMA for parameter-efficient fine-tuning."""
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
            lora_linear.to(self.device)
            _set_module(self.llama, name, lora_linear)

        self._log(f"[MOMENTQAHead] LoRA applied to {len(replacements)} layers", "info")

    def fuse_channels(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Per-patch cross-channel attention fusion.

        Independently processes each patch position, pooling D=18 channels
        into a single token using a learnable query.

        Args:
            embeddings: (B, D=18, P=64, F) MOMENT embeddings

        Returns:
            fused: (B, P=64, F) channel-fused tokens
        """
        B, D, P, F = embeddings.shape

        # Rearrange to process each patch independently
        # (B, D, P, F) → (B, P, D, F) → (B*P, D, F)
        tokens = embeddings.permute(0, 2, 1, 3)  # (B, P, D, F)
        tokens_flat = tokens.reshape(B * P, D, F)  # (B*P, D, F)

        # Learnable query for channel pooling: (1, 1, F) → (B*P, 1, F)
        query = self.channel_query.expand(B * P, -1, -1)
        query = self.query_norm(query)

        # Cross-attention: query attends to all D channels
        # Query: (B*P, 1, F), Key/Value: (B*P, D, F) → Output: (B*P, 1, F)
        pooled, _ = self.channel_attn(
            query.float(),
            tokens_flat.float(),
            tokens_flat.float()
        )
        pooled = pooled.to(query.dtype)

        # Reshape back and apply normalization
        # (B*P, 1, F) → (B, P, F)
        fused = pooled.view(B, P, F)
        fused = self.channel_norm(fused)

        return fused

    def prepare_text_batch(
        self,
        questions: List[str],
        answers: List[str],
        num_sensor_tokens: int,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare text batch with sensor token placeholders.

        Args:
            questions: List of question strings
            answers: List of answer strings
            num_sensor_tokens: Number of sensor embeddings (64 for MOMENT)
            device: Device to create tensors on

        Returns:
            Dict with input_ids, attention_mask, labels, sensor_positions
        """
        batch_size = len(questions)
        max_len = 0
        prepared = []

        sensor_token_id = self.tokenizer.convert_tokens_to_ids("[SENSOR]")

        for i, (q, a) in enumerate(zip(questions, answers)):
            # Build sequence: "Question: {q} [SENSOR]×num_sensor_tokens Answer: {a}"
            prefix = f"Question: {q} "
            prefix_ids = self.tokenizer(prefix, add_special_tokens=False).input_ids

            # Sensor tokens (64 for MOMENT)
            sensor_ids = [sensor_token_id] * num_sensor_tokens

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
        moment_embeddings: torch.Tensor,
        questions: List[str],
        answers: List[str],
        pad_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass through MOMENT QA head.

        Args:
            moment_embeddings: (B, D=18, P=64, F) MOMENT encoder output
            questions: List of question strings
            answers: List of answer strings
            pad_mask: Optional (B, P) padding mask (not used for fixed-length MOMENT)

        Returns:
            loss: Scalar loss
            info: Dict with logits, labels, etc.
        """
        device = moment_embeddings.device
        B = moment_embeddings.shape[0]

        # Step 1: Per-patch cross-channel fusion
        # (B, D=18, P=64, F) → (B, P=64, F)
        fused = self.fuse_channels(moment_embeddings)

        # Step 2: Project to LLaMA space
        # (B, P=64, F) → (B, P=64, llama_dim)
        projected = self.projector(fused)

        # For MOMENT, all samples have 64 sensor tokens
        num_sensor_tokens = projected.shape[1]  # 64

        # Prepare text batch with [SENSOR] placeholders
        tokenized = self.prepare_text_batch(questions, answers, num_sensor_tokens, device)

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

            # Use all 64 sensor embeddings
            sensor_embeds = projected[i]  # (64, llama_hidden)

            # Inject (should always be exactly 64 positions)
            if sensor_embeds.size(0) == num_positions:
                embeds[i, positions, :] = sensor_embeds
            elif sensor_embeds.size(0) < num_positions:
                # Pad if somehow we have fewer (shouldn't happen)
                pad_len = num_positions - sensor_embeds.size(0)
                pad_tensor = torch.zeros((pad_len, sensor_embeds.size(-1)), device=device, dtype=sensor_embeds.dtype)
                sensor_embeds = torch.cat([sensor_embeds, pad_tensor], dim=0)
                embeds[i, positions, :] = sensor_embeds
            else:
                # Truncate if needed
                embeds[i, positions, :] = sensor_embeds[:num_positions]

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

    def forward_autoregressive(
        self,
        moment_embeddings: torch.Tensor,
        questions: List[str],
        answers: List[str],
        return_predictions: bool = False,
    ):
        """
        Autoregressive training: generate tokens using model's own predictions,
        then compute loss against ground truth.

        This avoids teacher forcing and forces the model to learn from sensor data.

        Args:
            moment_embeddings: (B, D=18, P=64, F) MOMENT encoder output
            questions: List of question strings
            answers: List of ground truth answers
            return_predictions: If True, return dict with loss and decoded predictions

        Returns:
            If return_predictions=False: Scalar cross-entropy loss
            If return_predictions=True: Dict with {"loss": loss, "predictions": List[str]}
        """
        device = moment_embeddings.device
        batch_size = len(questions)

        # Step 1: Channel fusion and projection
        fused = self.fuse_channels(moment_embeddings)  # (B, 64, F)
        projected = self.projector(fused)  # (B, 64, llama_hidden)

        num_sensor_tokens = projected.shape[1]  # Always 64 for MOMENT

        # Step 2: Prepare prompt embeddings (question + sensors + "Answer:")
        dummy_answers = [""] * batch_size
        tokenized_prompt = self.prepare_text_batch(questions, dummy_answers, num_sensor_tokens, device)

        # Inject sensor embeddings at [SENSOR] positions
        base_embeds = self.llama.get_input_embeddings()(tokenized_prompt["input_ids"])
        if base_embeds.dtype != projected.dtype:
            base_embeds = base_embeds.to(projected.dtype)
        prompt_embeds = base_embeds.clone()

        for i, positions in enumerate(tokenized_prompt["sensor_positions"]):
            num_positions = positions.numel()
            if num_positions == 0:
                continue

            sensor_embeds = projected[i]  # (64, llama_hidden)

            # Inject all 64 sensor tokens
            if sensor_embeds.size(0) == num_positions:
                prompt_embeds[i, positions, :] = sensor_embeds
            elif sensor_embeds.size(0) < num_positions:
                # Pad if needed
                pad_len = num_positions - sensor_embeds.size(0)
                pad_tensor = torch.zeros((pad_len, sensor_embeds.size(-1)), device=device, dtype=sensor_embeds.dtype)
                sensor_embeds = torch.cat([sensor_embeds, pad_tensor], dim=0)
                prompt_embeds[i, positions, :] = sensor_embeds
            else:
                # Truncate if needed
                prompt_embeds[i, positions, :] = sensor_embeds[:num_positions]

        prompt_length = prompt_embeds.shape[1]

        # Step 3: Tokenize ground truth answers
        answer_texts = [" " + ans + self.tokenizer.eos_token for ans in answers]
        answer_tokens = self.tokenizer(
            answer_texts,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt"
        ).input_ids.to(device)  # (B, max_answer_len)

        max_answer_length = answer_tokens.shape[1]

        # Step 4: Autoregressively generate tokens with gradient tracking
        current_embeds = prompt_embeds  # Start with prompt
        all_logits = []
        all_predicted_ids = [] if return_predictions else None

        embedding_layer = self.llama.get_input_embeddings()

        for step in range(max_answer_length):
            # Forward pass through LLaMA
            attention_mask = torch.ones(
                current_embeds.shape[:2],
                dtype=torch.long,
                device=device
            )

            outputs = self.llama(
                inputs_embeds=current_embeds,
                attention_mask=attention_mask,
                use_cache=False,  # Disable cache for training
            )

            # Get logits for next token position
            next_token_logits = outputs.logits[:, -1, :]  # (B, vocab_size)
            all_logits.append(next_token_logits)

            # Sample next token (greedy for deterministic training)
            next_token_ids = next_token_logits.argmax(dim=-1)  # (B,)

            # Collect predicted token IDs if requested
            if return_predictions:
                all_predicted_ids.append(next_token_ids)

            # Get embeddings for next token
            next_token_embeds = embedding_layer(next_token_ids.unsqueeze(1))  # (B, 1, hidden)

            # Append to sequence for next iteration
            current_embeds = torch.cat([current_embeds, next_token_embeds], dim=1)

        # Step 5: Stack all logits and compute cross-entropy loss
        logits = torch.stack(all_logits, dim=1)  # (B, max_answer_length, vocab_size)

        # Compute loss against ground truth answer tokens
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),  # (B*seq_len, vocab_size)
            answer_tokens.reshape(-1),             # (B*seq_len,)
            ignore_index=self.tokenizer.pad_token_id,
            reduction='mean'
        )

        # If predictions requested, decode token IDs to text
        if return_predictions:
            # Stack predicted tokens: (B, max_answer_length)
            predicted_ids = torch.stack(all_predicted_ids, dim=1)

            # Decode each sample's predicted tokens to text
            predictions = []
            for i in range(batch_size):
                pred_tokens = predicted_ids[i].detach().cpu().tolist()
                # Decode and skip special tokens
                pred_text = self.tokenizer.decode(pred_tokens, skip_special_tokens=True)
                predictions.append(pred_text)

            return {"loss": loss, "predictions": predictions}

        return loss

    def generate(
        self,
        moment_embeddings: torch.Tensor,
        questions: List[str],
        pad_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 32,
        temperature: float = 0.7,
        do_sample: bool = False,
    ) -> List[str]:
        """
        Generate answers autoregressively.

        Args:
            moment_embeddings: (B, D=18, P=64, F) MOMENT encoder output
            questions: List of question strings
            pad_mask: Optional padding mask
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to sample or use greedy

        Returns:
            List of generated answer strings
        """
        device = moment_embeddings.device
        B = moment_embeddings.shape[0]

        # Step 1: Channel fusion
        fused = self.fuse_channels(moment_embeddings)

        # Step 2: Project
        projected = self.projector(fused)

        num_sensor_tokens = projected.shape[1]  # 64

        # Prepare prompt (no answers)
        dummy_answers = [""] * B
        tokenized = self.prepare_text_batch(questions, dummy_answers, num_sensor_tokens, device)

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
            if sensor_embeds.size(0) == num_positions:
                embeds[i, positions, :] = sensor_embeds
            elif sensor_embeds.size(0) < num_positions:
                pad_len = num_positions - sensor_embeds.size(0)
                pad_tensor = torch.zeros((pad_len, sensor_embeds.size(-1)), device=device, dtype=sensor_embeds.dtype)
                sensor_embeds = torch.cat([sensor_embeds, pad_tensor], dim=0)
                embeds[i, positions, :] = sensor_embeds
            else:
                embeds[i, positions, :] = sensor_embeds[:num_positions]

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
        head_path = os.path.join(out_dir, f"moment_qa_head_e{epoch}.pt")
        torch.save(self.state_dict(), head_path)
        tokenizer_dir = os.path.join(out_dir, f"tokenizer_e{epoch}")
        self.tokenizer.save_pretrained(tokenizer_dir)
        self._log(f"[SAVE] QA head → {head_path}, tokenizer → {tokenizer_dir}", "info")

    def count_parameters(self):
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
        }
