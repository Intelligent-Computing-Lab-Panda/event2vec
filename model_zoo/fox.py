# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# https://github.com/fla-org/flash-linear-attention/blob/v0.3.2/fla/layers/forgetting_attn.py
from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers.utils import logging

from fla.layers.utils import pad_input, unpad_input
from fla.modules import GroupNorm
from fla.ops.attn.decoding import attn_decoding_one_step
from fla.ops.forgetting_attn.parallel import parallel_forgetting_attn
import warnings

if TYPE_CHECKING:
    from fla.models.utils import Cache

logger = logging.get_logger(__name__)

import torch
import torch.nn.functional as F


class ForgettingAttention(nn.Module):

    def __init__(
            self,
            hidden_size: int = 2048,
            num_heads: int = 32,
            num_kv_heads: Optional[int] = None,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            window_size: Optional[int] = None,
            use_output_gate: bool = False,
            layer_idx: int = None
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        if num_kv_heads is None:
            self.num_kv_heads = self.num_heads
        else:
            self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_heads // self.num_kv_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.qkv_bias = qkv_bias
        self.qk_norm = qk_norm

        self.window_size = window_size
        self.use_output_gate = use_output_gate
        self.layer_idx = layer_idx

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=self.qkv_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=self.qkv_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=self.qkv_bias)
        self.f_proj = nn.Linear(self.hidden_size, self.num_heads, bias=True)

        if use_output_gate:
            self.g_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        if qk_norm:
            self.q_norm = GroupNorm(
                num_groups=self.num_heads,
                hidden_size=self.hidden_size,
                is_rms_norm=True,
            )
            self.k_norm = GroupNorm(
                num_groups=self.num_kv_heads,
                hidden_size=self.kv_dim,
                is_rms_norm=True,
            )

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.LongTensor] = None,
            past_key_values = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        batch_size, q_len, _ = hidden_states.size()

        q, k, v = self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)
        f = F.logsigmoid(self.f_proj(hidden_states).float())
        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)

        cu_seqlens = kwargs.get('cu_seqlens', None)
        if past_key_values is not None:
            assert cu_seqlens is None, "cu_seqlens should not be provided when past_key_values is not None"
            state = past_key_values.update(
                attn_state=(k, v, f),
                layer_idx=self.layer_idx,
                offset=q_len,
                cache_kwargs=dict(window_size=self.window_size)
            )
            k, v, f = state['attn_state']

        q = rearrange(q, '... (h d) -> ... h d', d=self.head_dim)
        k = rearrange(k, '... (h d) -> ... h d', d=self.head_dim)
        v = rearrange(v, '... (h d) -> ... h d', d=self.head_dim)

        if attention_mask is not None:
            q, (k, v, f), indices_q, cu_seqlens, max_seq_lens = unpad_input(q, (k, v, f), attention_mask, q_len,
                                                                            keepdim=True)
            _, cu_seqlens_k = cu_seqlens
            cu_seqlens = cu_seqlens_k
            max_seqlen_q, max_seqlen_k = max_seq_lens
            if max_seqlen_q != max_seqlen_k:
                assert max_seqlen_q == 1, "only support q_len == 1 for decoding"
                o = attn_decoding_one_step(q, k, v, f, cu_seqlens=cu_seqlens)
            else:
                o = parallel_forgetting_attn(q, k, v, f, cu_seqlens=cu_seqlens)
        else:
            o = parallel_forgetting_attn(q, k, v, f, cu_seqlens=cu_seqlens)
        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices_q, batch_size, q_len)
        o = rearrange(o, '... h d -> ... (h d)')
        if self.use_output_gate:
            o = self.g_proj(hidden_states).sigmoid() * o
        o = self.o_proj(o)
        return o, None, past_key_values


def parallel_forgetting_attn_torch(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        scale: Optional[float] = None,
        cu_seqlens: Optional[torch.LongTensor] = None,
        head_first: bool = False
) -> torch.Tensor:
    r"""
    parallel_forgetting_attn 的纯 PyTorch 实现，支持变长序列 (cu_seqlens)。

    Args:
        q (torch.Tensor):
            查询张量, 形状为 `[B, T, HQ, K]`。
        k (torch.Tensor):
            键张量, 形状为 `[B, T, H, K]`。
            如果 HQ 可被 H 整除，将应用 GQA。
        v (torch.Tensor):
            值张量, 形状为 `[B, T, H, V]`。
        g (torch.Tensor):
            每个时间步的对数衰减因子 (**在对数空间中**), 形状为 `[B, T, HQ]`。
        scale (Optional[float]):
            注意力分数的缩放因子。
            如果未提供，默认为 `1 / sqrt(K)`。 默认: `None`。
        cu_seqlens (torch.LongTensor):
            用于变长训练的累积序列长度，形状为 `[S+1]`，S是序列数量。
            与 FlashAttention API 兼容。
        head_first (Optional[bool]):
            输入是否为 head-first 格式。此参数已弃用。默认: `False`。

    Returns:
        o (torch.Tensor):
            输出张量, 形状为 `[B, T, HQ, V]`。
    """
    # --- 参数和形状检查 ---
    if head_first:
        raise DeprecationWarning(
            "head_first is deprecated and will be removed in a future version. "
            "Please use head_first=False for now instead."
        )
    if cu_seqlens is not None:
        assert q.shape[0] == 1, "使用 cu_seqlens 时，批次大小必须为 1"

    # 警告：检查是否有潜在的 head-first/batch-first 格式混淆
    if not head_first and q.shape[1] < q.shape[2]:
        warnings.warn(
            f"输入张量形状可能存在格式不匹配: seq_len ({q.shape[1]}) < num_heads ({q.shape[2]}). "
            "这可能表明当 head_first=False 时，输入是以 head-first 格式 [B, H, T, ...] 传入的。"
            "请验证您的输入张量格式是否符合预期的 [B, T, H, ...] 形状。"
        )

    # --- 准备工作 ---
    # 将输入从 [B, T, H, D] 转换为 [B, H, T, D] 以便进行矩阵乘法
    q, k, v = map(lambda x: x.permute(0, 2, 1, 3), (q, k, v))

    # 支持 Grouped-Query Attention (GQA)
    if k.shape[1] != q.shape[1]:
        num_kv_groups = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(num_kv_groups, dim=1)
        v = v.repeat_interleave(num_kv_groups, dim=1)

    if scale is None:
        scale = q.shape[-1] ** -0.5

    # --- 注意力计算 ---
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    # 应用遗忘机制 (forgetting mechanism)
    # g 的形状是 [B, T, H_q], 需要转换为 [B, H_q, T]
    g = g.permute(0, 2, 1)
    g_cumsum = torch.cumsum(g, dim=-1)
    # 通过广播创建遗忘矩阵 G_ij = g_cumsum_i - g_cumsum_j
    # [B, H, T, 1] - [B, H, 1, T] -> [B, H, T, T]
    g_matrix = g_cumsum.unsqueeze(-1) - g_cumsum.unsqueeze(-2)
    scores = scores + g_matrix

    # --- 掩码应用 ---
    seq_len = q.shape[2]
    # 始终应用 Causal Mask
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool), diagonal=1)

    # 如果提供了 cu_seqlens，则创建并应用 block-diagonal mask
    if cu_seqlens is not None:
        cu_seqlens = cu_seqlens.to(device=q.device)
        seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]

        # 创建一个序列 ID 张量，例如 [0, 0, 0, 1, 1, 2, 2, 2, 2, ...]
        indices = torch.arange(len(seq_lens), device=q.device)
        seq_ids = torch.repeat_interleave(indices, seq_lens)

        # 创建一个掩码，用于屏蔽不同序列之间的注意力
        inter_seq_mask = seq_ids.unsqueeze(1) != seq_ids.unsqueeze(0)
        final_mask = causal_mask | inter_seq_mask
    else:
        final_mask = causal_mask

    scores.masked_fill_(final_mask, -torch.inf)

    # --- 计算输出 ---
    # 使用 float32 计算 softmax 以保证数值稳定性
    attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    output = torch.matmul(attn_weights, v)  # 形状: [B, H, T, D]

    # 将输出转换回 [B, T, H, D] 格式
    return output.permute(0, 2, 1, 3)


class TorchGroupNorm(nn.Module):
    def __init__(
            self,
            num_groups: int,
            hidden_size: int,
            is_rms_norm: bool = False,
            eps: float = 1e-5,
            elementwise_affine: bool = True
    ):
        super().__init__()

        if hidden_size % num_groups != 0:
            raise ValueError('hidden_size must be divisible by num_groups')

        self.num_groups = num_groups
        self.hidden_size = hidden_size
        self.is_rms_norm = is_rms_norm
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        else:
            self.register_parameter('weight', None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            nn.init.ones_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (..., hidden_size)
        original_shape = x.shape
        # Reshape to (..., num_groups, group_dim) for group-wise normalization
        x = x.view(-1, self.num_groups, self.hidden_size // self.num_groups)

        if self.is_rms_norm:
            # RMSNorm logic: x / sqrt(mean(x^2) + eps)
            variance = x.pow(2).mean(dim=-1, keepdim=True)
            x_normalized = x * torch.rsqrt(variance + self.eps)
        else:
            # Standard GroupNorm/LayerNorm logic: (x - mean(x)) / sqrt(var(x) + eps)
            mean = x.mean(dim=-1, keepdim=True)
            variance = x.var(dim=-1, keepdim=True, unbiased=False)
            x_normalized = (x - mean) * torch.rsqrt(variance + self.eps)

        # Reshape back to original hidden_size dimension
        x_normalized = x_normalized.view(*original_shape)

        if self.elementwise_affine:
            x_normalized = x_normalized * self.weight

        return x_normalized

class BiForgettingAttention(nn.Module):

    def __init__(
            self,
            hidden_size: int = 2048,
            num_heads: int = 32,
            num_kv_heads: Optional[int] = None,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            window_size: Optional[int] = None,
            use_output_gate: bool = False,
            layer_idx: int = None,
            backend='triton'
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        if num_kv_heads is None:
            self.num_kv_heads = self.num_heads
        else:
            self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_heads // self.num_kv_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.qkv_bias = qkv_bias
        self.qk_norm = qk_norm

        self.window_size = window_size
        self.use_output_gate = use_output_gate
        self.layer_idx = layer_idx

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=self.qkv_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=self.qkv_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=self.qkv_bias)
        self.f_proj = nn.Linear(self.hidden_size, self.num_heads, bias=True)

        if use_output_gate:
            self.g_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)

        if qk_norm:
            if backend == 'triton':

                self.q_norm = GroupNorm(
                    num_groups=self.num_heads,
                    hidden_size=self.hidden_size,
                    is_rms_norm=True,
                )
                self.k_norm = GroupNorm(
                    num_groups=self.num_kv_heads,
                    hidden_size=self.kv_dim,
                    is_rms_norm=True,
                )
            else:
                self.q_norm = TorchGroupNorm(
                    num_groups=self.num_heads,
                    hidden_size=self.hidden_size,
                    is_rms_norm=True,
                )
                self.k_norm = TorchGroupNorm(
                    num_groups=self.num_kv_heads,
                    hidden_size=self.kv_dim,
                    is_rms_norm=True,
                )


        self.backend = backend
        if backend != 'triton':
            assert backend == 'torch'
            warnings.warn('Note that torch backend is only used for experiments!')
    # (This is the corrected method for your BiForgettingAttention class)

    def get_bidirectional_attention_matrices(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.LongTensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes and returns the forward and backward attention matrices for visualization.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (forward_attention, backward_attention)
        """
        # --- Setup code (remains the same) ---
        q, k = self.q_proj(hidden_states), self.k_proj(hidden_states)
        f = self.f_proj(hidden_states).float()
        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)

        # Encapsulate the unidirectional computation logic to avoid code duplication
        def _compute_unidirectional_attn(q_in, k_in, f_in, mask_in):
            q_in = rearrange(q_in, 'b t (h d) -> b h t d', h=self.num_heads)
            k_in = rearrange(k_in, 'b t (h d) -> b h t d', h=self.num_kv_heads)

            # ✅ START: THE FIX FOR GROUPED-QUERY ATTENTION
            # Repeat K heads to match Q heads
            if self.num_kv_groups > 1:
                k_in = k_in.repeat_interleave(self.num_kv_groups, dim=1)
            # ✅ END: THE FIX

            f_in = f_in.permute(0, 2, 1)
            g_cumsum = torch.cumsum(f_in, dim=-1)

            # Now the matrix multiplication will work correctly
            scores = torch.matmul(q_in, k_in.transpose(-2, -1))
            scores = scores + g_cumsum.unsqueeze(-1) - g_cumsum.unsqueeze(-2)

            seq_len = q_in.size(2)
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool), diagonal=1)
            scores.masked_fill_(causal_mask, float('-inf'))

            if mask_in is not None:
                mask = mask_in.unsqueeze(1).unsqueeze(2)
                scores.masked_fill_(~mask.bool(), float('-inf'))

            return F.softmax(scores, dim=-1)

        # 1. Compute the forward attention matrix
        forward_attn = _compute_unidirectional_attn(q, k, f, attention_mask)

        # 2. Compute the backward attention matrix (on the flipped sequence)
        q_rev = torch.flip(q, dims=[1])
        k_rev = torch.flip(k, dims=[1])
        f_rev = torch.flip(f, dims=[1])
        mask_rev = torch.flip(attention_mask, dims=[1]) if attention_mask is not None else None

        backward_attn_rev = _compute_unidirectional_attn(q_rev, k_rev, f_rev, mask_rev)

        # 3. Flip the backward attention matrix back to align with the original sequence
        backward_attn = torch.flip(backward_attn_rev, dims=[-2, -1])

        return forward_attn, backward_attn

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.LongTensor] = None,
            past_key_values = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        batch_size, q_len, _ = hidden_states.size()

        q, k, v = self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)
        f = F.logsigmoid(self.f_proj(hidden_states).float())
        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)

        cu_seqlens = kwargs.get('cu_seqlens', None)
        if past_key_values is not None:
            assert cu_seqlens is None, "cu_seqlens should not be provided when past_key_values is not None"
            state = past_key_values.update(
                attn_state=(k, v, f),
                layer_idx=self.layer_idx,
                offset=q_len,
                cache_kwargs=dict(window_size=self.window_size)
            )
            k, v, f = state['attn_state']

        q = rearrange(q, '... (h d) -> ... h d', d=self.head_dim)
        k = rearrange(k, '... (h d) -> ... h d', d=self.head_dim)
        v = rearrange(v, '... (h d) -> ... h d', d=self.head_dim)

        assert attention_mask is not None
        if self.use_output_gate:
            g = self.g_proj(hidden_states).sigmoid()
        else:
            g = None
        outputs = []
        for q, k, v, f, g, attention_mask in (
                (q, k, v, f, g, attention_mask),
                (torch.flip(q, dims=[1]), torch.flip(k, dims=[1]), torch.flip(v, dims=[1]), torch.flip(f, dims=[1]),
                 None if g is None else torch.flip(g, dims=[1]), torch.flip(attention_mask, dims=[1])),
        ):
            if attention_mask is not None:
                q, (k, v, f), indices_q, cu_seqlens, max_seq_lens = unpad_input(q, (k, v, f), attention_mask, q_len,
                                                                                keepdim=True)
                _, cu_seqlens_k = cu_seqlens
                cu_seqlens = cu_seqlens_k
                max_seqlen_q, max_seqlen_k = max_seq_lens
                if max_seqlen_q != max_seqlen_k:
                    assert max_seqlen_q == 1, "only support q_len == 1 for decoding"
                    o = attn_decoding_one_step(q, k, v, f, cu_seqlens=cu_seqlens)
                else:
                    if q.get_device() == -1 or self.backend == 'torch':
                        o = parallel_forgetting_attn_torch(q, k, v, f, cu_seqlens=cu_seqlens)
                    else:
                        o = parallel_forgetting_attn(q, k, v, f, cu_seqlens=cu_seqlens)
            else:
                if q.get_device() == -1 or self.backend == 'torch':
                    o = parallel_forgetting_attn_torch(q, k, v, f, cu_seqlens=cu_seqlens)
                else:
                    o = parallel_forgetting_attn(q, k, v, f, cu_seqlens=cu_seqlens)
            if attention_mask is not None:
                o = pad_input(o.squeeze(0), indices_q, batch_size, q_len)
            o = rearrange(o, '... h d -> ... (h d)')
            if self.use_output_gate:
                o = g * o
            outputs.append(o)

        outputs[1] = torch.flip(outputs[1], dims=[1])
        o = torch.cat(outputs, dim=2)
        o = self.o_proj(o)
        return o, None, past_key_values


# 3. 对比测试函数
def compare_cpu_gpu_outputs():
    """
    初始化 BiForgettingAttention 模型，分别在 CPU 和 GPU 上运行，并比较输出差异。
    """
    # --- 模型和输入参数定义 ---
    BATCH_SIZE = 2
    SEQ_LEN = 1024
    HIDDEN_SIZE = 1024
    NUM_HEADS = 16
    NUM_KV_HEADS = 4  # 使用 GQA

    # --- 初始化模型 ---
    # 使用相同的随机种子确保 CPU 和 GPU 模型的权重一致
    torch.manual_seed(42)
    model = BiForgettingAttention(
        hidden_size=HIDDEN_SIZE,
        num_heads=NUM_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        qk_norm=True, use_output_gate=False, backend='torch'
    )
    model.eval()  # 设置为评估模式
    sd = model.state_dict()

    # --- 准备 CPU 输入并运行 ---
    print("--- 1. Running on CPU ---")
    cpu_device = torch.device("cpu")
    model.to(cpu_device)

    # 使用相同的随机种子确保输入数据一致
    torch.manual_seed(123)
    hidden_states_cpu = torch.randn(
        BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE, device=cpu_device, dtype=torch.float32
    )

    print("Running PyTorch forward pass on CPU...")
    with torch.no_grad():
        output_cpu = \
        model(hidden_states_cpu, attention_mask=torch.ones([hidden_states_cpu.shape[0], hidden_states_cpu.shape[1]]))[0]
    print("CPU forward pass completed.")
    print(f"Output shape on CPU: {output_cpu.shape}\n")

    print("--- 2. Running on GPU ---")
    model = BiForgettingAttention(
        hidden_size=HIDDEN_SIZE,
        num_heads=NUM_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        qk_norm=True, use_output_gate=False, backend='triton'
    )
    model.load_state_dict(sd)
    gpu_device = torch.device("cuda")
    # 为了公平比较，使用 bfloat16 以匹配 Triton 内核通常使用的数据类型
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32

    model.to(gpu_device).to(dtype)
    hidden_states_gpu = hidden_states_cpu.to(gpu_device).to(dtype)

    print(f"Running Triton forward pass on GPU with dtype: {dtype}...")
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=dtype):
        output_gpu = model(hidden_states_gpu,
                           attention_mask=torch.ones([hidden_states_cpu.shape[0], hidden_states_cpu.shape[1]],
                                                     device=gpu_device))[0]
    print("GPU forward pass completed.")
    print(f"Output shape on GPU: {output_gpu.shape}\n")

    # --- 对比结果 ---
    print("--- 3. Comparison ---")
    # 将 GPU 输出移回 CPU 并转换为 float32 以进行比较
    output_gpu_cpu = output_gpu.to(cpu_device, dtype=torch.float32)

    max_diff = (output_cpu - output_gpu_cpu).abs().max().item()
    print(f"Max absolute difference between CPU and GPU outputs: {max_diff:.6f}")

    is_close = torch.allclose(output_cpu, output_gpu_cpu, atol=1e-2, rtol=1e-2)
    print(f"Are outputs close (torch.allclose with atol=1e-2, rtol=1e-2)? {is_close}")


