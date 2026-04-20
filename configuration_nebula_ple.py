# coding=utf-8
"""Nebula-PLE configuration (Qwen3-compatible with per-layer embeddings)."""

from transformers.models.qwen3.configuration_qwen3 import Qwen3Config


class NebulaPLEConfig(Qwen3Config):
    model_type = "nebula_ple"

    def __init__(
        self,
        hidden_size_per_layer_input: int = 0,
        vocab_size_per_layer_input: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size_per_layer_input = int(hidden_size_per_layer_input or 0)
        self.vocab_size_per_layer_input = (
            int(vocab_size_per_layer_input)
            if vocab_size_per_layer_input is not None
            else int(self.vocab_size)
        )


__all__ = ["NebulaPLEConfig"]
