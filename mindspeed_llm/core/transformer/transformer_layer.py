# coding=utf-8
# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import Any, Dict, Optional, Tuple
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F

from megatron.core import tensor_parallel
from megatron.core.transformer.transformer_layer import TransformerLayerSubmodules
from megatron.core.utils import WrappedTensor, deprecate_inference_params
from megatron.core.transformer.transformer_layer import TransformerLayer as MegatronTransformerLayer
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.experts import GroupedMLP, SequentialMLP
from megatron.core.utils import make_viewless_tensor
from megatron.training import get_args


_MEKI_GATE_STATS = {}


def record_meki_gate_stats(layer_number: int, gate: Tensor):
    """Collect detached MeKi gate scalar stats for training-time logging."""
    if gate.numel() == 0:
        return
    with torch.no_grad():
        gate = gate.detach().float()
        layer_stats = _MEKI_GATE_STATS.setdefault(
            int(layer_number),
            {"mean": [], "std": [], "min": [], "max": []},
        )
        layer_stats["mean"].append(gate.mean())
        layer_stats["std"].append(gate.std(unbiased=False))
        layer_stats["min"].append(gate.min())
        layer_stats["max"].append(gate.max())


def pop_meki_gate_stats():
    """Return averaged MeKi gate stats and clear the collection buffer."""
    stats = {}
    for layer_number, layer_stats in _MEKI_GATE_STATS.items():
        stats[layer_number] = {}
        for name, values in layer_stats.items():
            if values:
                stats[layer_number][name] = torch.stack(values).mean().item()
    _MEKI_GATE_STATS.clear()
    return stats


class TransformerLayer(MegatronTransformerLayer):
    """
    Inherited from megatron TransformerLayer.
    """

    def __init__(
            self,
            config: TransformerConfig,
            submodules: TransformerLayerSubmodules,
            layer_number: int = 1,
            hidden_dropout: float = None,
    ):
        super().__init__(config=config,
                         submodules=submodules,
                         layer_number=layer_number,
                         hidden_dropout=hidden_dropout)

        # For mcore activation re-computation
        if self.mlp.__class__ is MoELayer:
            if isinstance(self.mlp.experts, GroupedMLP):
                self.mlp.experts.layer_number = self.layer_number
            if self.mlp.experts.__class__ is SequentialMLP:
                for expert in self.mlp.experts.local_experts:
                    expert.layer_number = self.layer_number
        else:
            self.mlp.layer_number = self.layer_number
        # set mtp_idx
        args = get_args()
        if args.mtp_num_layers and hasattr(self.self_attention, "core_attention"):
            self.mtp_idx = 0
            self.self_attention.core_attention.mtp_idx = 0

        self.hidden_size_per_layer_input = int(getattr(config, "hidden_size_per_layer_input", 0) or 0)
        if self.hidden_size_per_layer_input > 0:
            self.ple_alpha = float(getattr(config, "ple_alpha", 0.1))
            self.ple_act = nn.GELU()
            self.per_layer_input_gate = nn.Linear(self.config.hidden_size, self.hidden_size_per_layer_input, bias=False)
            self.per_layer_projection = nn.Linear(self.hidden_size_per_layer_input, self.config.hidden_size, bias=False)
            self.post_per_layer_input_norm = nn.LayerNorm(self.config.hidden_size, eps=self.config.layernorm_epsilon)

        self.meki_dim = int(getattr(config, "meki_dim", 0) or 0)
        self.use_meki = self.meki_dim > 0
        self.meki_fusion_mode = str(getattr(config, "meki_fusion_mode", "ple_gelu_mul")).lower()
        self._meki_input = None
        self._meki_input_ids = None
        self._meki_word_emb = None
        if self.use_meki:
            self.meki_gate_proj = nn.Linear(self.config.hidden_size, self.meki_dim, bias=False)
            self.meki_out_proj = nn.Linear(self.meki_dim, self.config.hidden_size, bias=False)
            meki_intermediate = max(self.config.hidden_size // 2, self.meki_dim)
            self.meki_word_gate_proj = nn.Linear(self.config.hidden_size, meki_intermediate, bias=False)
            self.meki_word_up_proj = nn.Linear(self.config.hidden_size, meki_intermediate, bias=False)
            self.meki_word_down_proj = nn.Linear(meki_intermediate, self.meki_dim, bias=False)
            vocab_size = int(
                getattr(self.config, "vocab_size", getattr(self.config, "padded_vocab_size", 0))
                or getattr(self.config, "padded_vocab_size", 0)
            )
            if vocab_size <= 0:
                raise RuntimeError("MeKi enabled but vocab_size is not configured.")
            self.meki_embeddings = nn.Embedding(vocab_size, self.meki_dim)
            self.meki_mix_norm = nn.RMSNorm(self.meki_dim, eps=self.config.layernorm_epsilon)
            self.meki_post_norm = nn.RMSNorm(self.config.hidden_size, eps=self.config.layernorm_epsilon)
            self.meki_alpha_scale = nn.Parameter(torch.tensor(float(getattr(config, "meki_alpha", 1.0))))
            self.meki_beta_scale = nn.Parameter(torch.tensor(float(getattr(config, "meki_beta", 1.0))))

    def _forward_attention(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        inference_context: Optional[Any] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[Tensor] = None,
        *,
        inference_params: Optional[Any] = None,
    ):
        """
        Perform a forward pass through the attention layer and the layernorms before and after
        the attention operations.

        Args:
            hidden_states (Tensor): Input tensor of shape [s, b, h] where s is sequence length,
                b is batch size, and h is hidden size.
            attention_mask (Tensor): Mask tensor for self-attention.
            context (Tensor, optional): Context tensor for cross-attention.
            context_mask (Tensor, optional): Mask tensor for cross-attention.
            rotary_pos_emb (Tensor, optional): Rotary positional embeddings.
            attention_bias (Tensor, optional): Bias tensor for Q * K.T.
            inference_context (object, optional): Parameters for inference-time optimizations.
            packed_seq_params (object, optional): Parameters for packed sequence processing.
            sequence_len_offset (Tensor, optional): Offset along sequence dimension
                during inference.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: A tuple containing:
                pre_mlp_layernorm_output (Tensor): Transformed hidden states before the MLP.
                residual (Tensor): Residual connection.
                context (Tensor): Updated context tensor if cross-attention is used,
                otherwise None.
        """
        args = get_args()
        inference_context = deprecate_inference_params(inference_context, inference_params)

        # Residual connection.
        residual = hidden_states

        # Optional Input Layer norm
        if self.recompute_input_layernorm:
            self.input_layernorm_checkpoint = tensor_parallel.CheckpointWithoutOutput()
            input_layernorm_output = self.input_layernorm_checkpoint.checkpoint(
                self.input_layernorm, hidden_states
            )
        else:
            input_layernorm_output = self.input_layernorm(hidden_states)

        # Self attention.
        attention_output_with_bias = self.self_attention(
            input_layernorm_output,
            attention_mask=attention_mask,
            inference_context=inference_context,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
        )

        # For minicpm model
        if args.scale_depth is not None:
            attention_output, attention_bias = attention_output_with_bias
            attention_output = attention_output * (args.scale_depth / math.sqrt(args.num_layers))
            attention_output_with_bias = (attention_output, attention_bias)

        if self.recompute_input_layernorm:
            # discard the output of the input layernorm and register the recompute
            # as a gradient hook of attention_output_with_bias[0]
            self.input_layernorm_checkpoint.discard_output_and_register_recompute(
                attention_output_with_bias[0]
            )

        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.self_attn_bda(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias, residual, self.hidden_dropout
            )

        # Residual connection.
        residual = hidden_states

        # Optional Layer norm after self-attention
        pre_cross_attn_layernorm_output = self.pre_cross_attn_layernorm(hidden_states)

        # Cross attention.
        attention_output_with_bias = self.cross_attention(
            pre_cross_attn_layernorm_output,
            attention_mask=context_mask,
            key_value_states=context,
            inference_context=inference_context,
        )

        if isinstance(attention_output_with_bias, dict) and "context" in attention_output_with_bias:
            context = attention_output_with_bias["context"]

        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.cross_attn_bda(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias, residual, self.hidden_dropout
            )

        # Residual connection.
        residual = hidden_states

        # Optional Layer norm post the cross-attention.
        if self.recompute_pre_mlp_layernorm:
            self.pre_mlp_norm_checkpoint = tensor_parallel.CheckpointWithoutOutput()
            pre_mlp_layernorm_output = self.pre_mlp_norm_checkpoint.checkpoint(
                self.pre_mlp_layernorm, hidden_states
            )
        else:
            pre_mlp_layernorm_output = self.pre_mlp_layernorm(hidden_states)

        return pre_mlp_layernorm_output, residual, context

    def _forward_mlp(self, pre_mlp_layernorm_output, residual):
        args = get_args()
        # MLP.
        if self.recompute_mlp:
            mlp_output_with_bias = tensor_parallel.checkpoint(
                self.mlp, False, pre_mlp_layernorm_output
            )
        else:
            mlp_output_with_bias = self.mlp(pre_mlp_layernorm_output)

        if self.recompute_pre_mlp_layernorm:
            # discard the output of the pre-mlp layernorm and register the recompute
            # as a gradient hook of mlp_output_with_bias[0]
            self.pre_mlp_norm_checkpoint.discard_output_and_register_recompute(
                mlp_output_with_bias[0]
            )

        if args.scale_depth is not None:
            mlp_output, mlp_bias = mlp_output_with_bias
            mlp_output = mlp_output * (args.scale_depth / math.sqrt(args.num_layers))
            mlp_output_with_bias = (mlp_output, mlp_bias)

        if self.use_meki:
            if self._meki_input_ids is None or self._meki_word_emb is None:
                raise RuntimeError(f"MeKi enabled but auxiliary inputs are missing at layer {self.layer_number}.")
            static_embedding = self.meki_embeddings(self._meki_input_ids).permute(1, 0, 2).contiguous()
            dynamic_hidden = F.silu(self.meki_word_gate_proj(self._meki_word_emb)) * self.meki_word_up_proj(
                self._meki_word_emb
            )
            dynamic_embedding = self.meki_word_down_proj(dynamic_hidden)
            meki_embedding = self.meki_mix_norm(static_embedding + dynamic_embedding * self.meki_beta_scale)
            meki_embedding = meki_embedding * self.meki_alpha_scale
            if self._meki_input is not None:
                meki_embedding = self.meki_mix_norm(self._meki_input + dynamic_embedding * self.meki_beta_scale)
                meki_embedding = meki_embedding * self.meki_alpha_scale
            if self.meki_fusion_mode == "meki_sigmoid_add":
                meki_gate = torch.sigmoid(self.meki_gate_proj(pre_mlp_layernorm_output))
                meki_fused = meki_gate + meki_embedding
            else:
                meki_gate = F.gelu(self.meki_gate_proj(pre_mlp_layernorm_output))
                meki_fused = meki_gate * meki_embedding
            record_meki_gate_stats(self.layer_number, meki_gate)
            meki_output = self.meki_out_proj(meki_fused)
            meki_output = self.meki_post_norm(meki_output)
            mlp_output_with_bias = (
                mlp_output_with_bias[0] + meki_output,
                mlp_output_with_bias[1],
            )
        self._meki_input = None
        self._meki_input_ids = None
        self._meki_word_emb = None

        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.mlp_bda(self.training, self.config.bias_dropout_fusion)(
                mlp_output_with_bias, residual, self.hidden_dropout
            )

        # Jit compiled function creates 'view' tensor. This tensor
        # potentially gets saved in the MPU checkpoint function context,
        # which rejects view tensors. While making a viewless tensor here
        # won't result in memory savings (like the data loader, or
        # p2p_communication), it serves to document the origin of this
        # 'view' tensor.
        output = make_viewless_tensor(
            inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
        )

        return output

    def apply_per_layer_input(self, hidden_states: Tensor, per_layer_input: Optional[Tensor]) -> Tensor:
        """Inject per-layer embedding signal after the base transformer layer output."""
        if self.hidden_size_per_layer_input <= 0:
            return hidden_states
        if per_layer_input is None:
            raise RuntimeError(f"PLE enabled but per_layer_input is None at layer {self.layer_number}.")

        residual = hidden_states
        gate = self.ple_act(self.per_layer_input_gate(hidden_states))
        hidden_states = gate * per_layer_input
        hidden_states = self.per_layer_projection(hidden_states)
        hidden_states = self.post_per_layer_input_norm(hidden_states)
        hidden_states = residual + hidden_states * self.ple_alpha
        return hidden_states

    def set_meki_input(self, meki_input: Optional[Tensor]):
        """Set layer-specific MeKi input before running the MLP branch."""
        self._meki_input = meki_input

    def set_meki_aux_inputs(self, meki_input_ids: Optional[Tensor], meki_word_emb: Optional[Tensor]):
        """Set MeKi auxiliary inputs shared across layers."""
        self._meki_input_ids = meki_input_ids
        self._meki_word_emb = meki_word_emb
