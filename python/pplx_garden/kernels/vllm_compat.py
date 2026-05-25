import os
from typing import Optional

import torch

from pplx_garden.distributed import ParallelGroup
from pplx_garden.kernels.p2p_all_to_all import P2PAllToAll
from pplx_garden.utils.math import ceil_div


def _env_value(name: str) -> Optional[str]:
    value = os.getenv(name)
    if value is None:
        value = os.getenv(f"VLLM_{name}")
    return value


def _env_bool(name: str, default: bool = False) -> bool:
    value = _env_value(name)
    if value is None or value == "":
        return default
    return value.lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: Optional[int]) -> Optional[int]:
    value = _env_value(name)
    if value is None or value == "":
        return default
    if value.lower() in ("none", "null"):
        return None
    return int(value)


def _out_dtype_from_env(default: torch.dtype) -> torch.dtype:
    value = (_env_value("PPLX_GARDEN_OUT_DTYPE") or "").lower()
    if value in ("", "auto"):
        return default

    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if value not in mapping:
        raise ValueError(f"Unsupported PPLX_GARDEN_OUT_DTYPE={value!r}")
    return mapping[value]


def _as_int32_bound(bound_m: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if bound_m is None or bound_m.dtype == torch.int32:
        return bound_m
    if bound_m.dtype == torch.uint32:
        return bound_m.view(torch.int32)
    return bound_m.to(torch.int32)


def _flatten_expert_tensor(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if tensor.dim() == 2:
        return tensor
    if tensor.dim() != 3:
        raise ValueError(f"{name} must be 2D or 3D, got {tensor.shape}")
    try:
        return tensor.view(tensor.shape[0] * tensor.shape[1],
                           tensor.shape[2])
    except RuntimeError as exc:
        raise ValueError(
            f"{name} must be contiguous in expert-major layout") from exc


def _debug_sync_and_check_counts(
    *,
    label: str,
    out_expert_num_tokens: torch.Tensor,
    flat_expert_x: torch.Tensor,
) -> None:
    if not _env_bool("PPLX_GARDEN_DEBUG_SYNC"):
        return

    torch.cuda.synchronize(flat_expert_x.device)
    num_experts = out_expert_num_tokens.numel()
    if num_experts == 0:
        return

    capacity = flat_expert_x.size(0) // num_experts
    counts = out_expert_num_tokens.detach().to("cpu", non_blocking=False)
    max_count = int(counts.max().item()) if counts.numel() else 0
    print(
        "PPLX_GARDEN_DEBUG "
        f"{label}: rank={torch.distributed.get_rank() if torch.distributed.is_initialized() else -1} "
        f"counts={counts.tolist()} capacity={capacity} max_count={max_count}",
        flush=True,
    )
    if max_count > capacity:
        raise RuntimeError(
            "pplx-garden dispatch produced more tokens for an expert than "
            f"the vLLM batched expert buffer can hold: max_count={max_count}, "
            f"capacity={capacity}, counts={counts.tolist()}")


class PplxKernelsCompatAllToAll:
    """pplx-kernels compatible facade over Garden P2PAllToAll.

    vLLM's proven PPLX MoE path expects the pplx-kernels AllToAll API. Garden
    needs explicit groups and uses scale element counts rather than scale bytes;
    this adapter keeps those differences inside the Garden package.
    """

    def __init__(
        self,
        *,
        max_num_tokens: int,
        num_experts: int,
        rank: int,
        world_size: int,
        hidden_dim: int,
        dp_size: int = 1,
        tp_size: Optional[int] = None,
        experts_per_token: Optional[int] = None,
        num_experts_per_token: Optional[int] = None,
        expert_padding: int = 1,
        hidden_dim_bytes: Optional[int] = None,
        hidden_dim_scale_bytes: Optional[int] = None,
        hidden_dim_scale: Optional[int] = None,
        in_dtype: Optional[torch.dtype] = None,
        out_dtype: Optional[torch.dtype] = None,
        scale_dtype: Optional[torch.dtype] = None,
        nets_per_gpu: int = 1,
        max_private_tokens: Optional[int] = None,
        expert_token_capacity: Optional[int] = None,
        device: Optional[torch.device] = None,
        global_group: ParallelGroup,
        dp_group: Optional[ParallelGroup],
        node_group: Optional[ParallelGroup],
    ) -> None:
        self.max_num_tokens = int(max_num_tokens)
        self.num_experts = int(num_experts)
        if num_experts_per_token is None:
            num_experts_per_token = experts_per_token
        if num_experts_per_token is None:
            raise ValueError("num_experts_per_token must be set")
        self.experts_per_token = int(num_experts_per_token)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.dp_size = int(dp_size)
        self.tp_size = int(tp_size) if tp_size is not None else self.dp_size
        self.hidden_dim = int(hidden_dim)
        self.expert_padding = int(expert_padding)
        self.hidden_dim_bytes = (
            None if hidden_dim_bytes is None else int(hidden_dim_bytes))
        self.hidden_dim_scale_bytes = (
            None if hidden_dim_scale_bytes is None else
            int(hidden_dim_scale_bytes))
        self.hidden_dim_scale = (
            None if hidden_dim_scale is None else int(hidden_dim_scale))
        self.in_dtype = in_dtype
        self.out_dtype = out_dtype
        self.scale_dtype = scale_dtype
        self.nets_per_gpu = int(nets_per_gpu)
        self.max_private_tokens = max_private_tokens
        self.expert_token_capacity = (
            None if expert_token_capacity is None else int(expert_token_capacity))
        self.device = device
        self.global_group = global_group
        self.dp_group = dp_group
        self.node_group = node_group
        self.compat_mode = (
            _env_value("PPLX_GARDEN_COMPAT_MODE") or "mask").lower()
        if self.compat_mode not in ("mask", "native"):
            raise ValueError(
                "PPLX_GARDEN_COMPAT_MODE must be 'mask' or 'native', "
                f"got {self.compat_mode!r}")
        self.mask_style = (
            _env_value("PPLX_GARDEN_MASK_STYLE") or "token_lane").lower()
        if self.mask_style not in ("token_lane", "expert_lane"):
            raise ValueError(
                "PPLX_GARDEN_MASK_STYLE must be 'token_lane' or "
                f"'expert_lane', got {self.mask_style!r}")

        self._handle: Optional[P2PAllToAll] = None
        self._weights_cache: dict[tuple[str, int, tuple[int, ...]],
                                  torch.Tensor] = {}

    def __del__(self) -> None:
        # P2PAllToAll.destroy() contains collectives; vLLM calls destroy()
        # explicitly while all ranks are still alive.
        self._handle = None

    def _weights_like(self, indices: torch.Tensor,
                      fill: float) -> torch.Tensor:
        device_index = indices.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        key = (str(fill), device_index, tuple(indices.shape))
        cached = self._weights_cache.get(key)
        if cached is None or cached.device != indices.device:
            cached = torch.full(indices.shape,
                                fill,
                                dtype=torch.float32,
                                device=indices.device)
            self._weights_cache[key] = cached
        return cached

    def _dummy_indices(self, indices: torch.Tensor) -> torch.Tensor:
        experts_per_rank = ceil_div(self.num_experts, self.world_size)
        dummy_expert = min(self.rank * experts_per_rank, self.num_experts - 1)
        return torch.full((1, self.experts_per_token),
                          dummy_expert,
                          dtype=torch.uint32,
                          device=indices.device)

    def _dummy_scale(
        self,
        scale: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if scale is None:
            return None
        if scale.dim() == 0:
            return scale
        shape = list(scale.shape)
        if shape[0] == 0:
            shape[0] = 1
        return torch.zeros(shape, dtype=scale.dtype, device=scale.device)

    def _mask_weights_for_lane(
        self,
        indices: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        if self.compat_mode != "mask" or self.tp_size <= 1:
            return weights

        if self.mask_style == "expert_lane":
            experts_per_rank = ceil_div(self.num_experts, self.world_size)
            dst_ranks = torch.div(indices.to(torch.int64),
                                  experts_per_rank,
                                  rounding_mode="floor")
            keep = (dst_ranks % self.tp_size) == (self.rank % self.tp_size)
        else:
            token_ids = torch.arange(indices.size(0),
                                     dtype=torch.int64,
                                     device=indices.device)
            keep = ((token_ids % self.tp_size) ==
                    (self.rank % self.tp_size)).view(-1, 1)
        return torch.where(keep, weights, torch.zeros_like(weights))

    def _ensure_handle(
        self,
        *,
        dp_x: torch.Tensor,
        dp_x_scale: Optional[torch.Tensor],
    ) -> P2PAllToAll:
        if self._handle is not None:
            return self._handle

        if self.global_group.rank != self.rank or self.global_group.size != self.world_size:
            raise RuntimeError(
                "pplx-garden rank mismatch: "
                f"global_group rank/size={self.global_group.rank}/{self.global_group.size}, "
                f"all2all args rank/size={self.rank}/{self.world_size}")
        if self.dp_group is None:
            if self.dp_size != 1:
                raise RuntimeError(
                    "pplx-garden dp_size mismatch: "
                    f"vLLM passed dp_size={self.dp_size}, "
                    "but no dp_group is configured")
        elif self.dp_size != self.dp_group.size:
            raise RuntimeError(
                "pplx-garden dp_size mismatch: "
                f"vLLM passed dp_size={self.dp_size}, "
                f"dp_group size={self.dp_group.size}")
        if self.compat_mode == "mask":
            if self.dp_size != 1:
                raise RuntimeError(
                    "pplx-garden mask compatibility mode requires dp_size=1; "
                    f"got dp_size={self.dp_size}")
            if self.tp_size > 1 and self.node_group is None:
                raise RuntimeError(
                    "pplx-garden mask compatibility mode requires a TP "
                    "node_group for the final all-reduce")

        scale_dtype = self.scale_dtype
        if scale_dtype is None and dp_x_scale is not None:
            scale_dtype = dp_x_scale.dtype

        hidden_dim_scale = self.hidden_dim_scale
        if hidden_dim_scale is None and self.hidden_dim_scale_bytes:
            if scale_dtype is None:
                raise RuntimeError(
                    "hidden_dim_scale_bytes is nonzero but dp_x_scale is None")
            hidden_dim_scale = self.hidden_dim_scale_bytes // scale_dtype.itemsize

        device = self.device
        if device is None:
            device = torch.device("cuda",
                                  dp_x.device.index
                                  or torch.cuda.current_device())

        self._handle = P2PAllToAll(
            max_num_tokens=self.max_num_tokens,
            num_experts=self.num_experts,
            expert_padding=_env_int("PPLX_GARDEN_EXPERT_PADDING",
                                    self.expert_padding) or 1,
            hidden_dim=self.hidden_dim,
            hidden_dim_scale=hidden_dim_scale,
            in_dtype=self.in_dtype or dp_x.dtype,
            out_dtype=self.out_dtype or _out_dtype_from_env(dp_x.dtype),
            scale_dtype=scale_dtype,
            num_experts_per_token=self.experts_per_token,
            nets_per_gpu=_env_int("PPLX_GARDEN_NETS_PER_GPU",
                                  self.nets_per_gpu) or 1,
            max_private_tokens=_env_int("PPLX_GARDEN_MAX_PRIVATE_TOKENS",
                                        self.max_private_tokens),
            device=device,
            dp_group=self.dp_group,
            node_group=(None if _env_bool("PPLX_GARDEN_DISABLE_NVLINK") else
                        self.node_group),
            global_group=self.global_group,
            expert_token_capacity=(self.expert_token_capacity
                                   if self.expert_token_capacity is not None else
                                   self.max_num_tokens *
                                   (self.world_size // self.dp_size)),
        )
        return self._handle

    def dispatch(
        self,
        out_expert_num_tokens: torch.Tensor,
        out_expert_x: torch.Tensor,
        out_expert_x_scale: Optional[torch.Tensor],
        dp_x: torch.Tensor,
        dp_x_scale: Optional[torch.Tensor],
        indices: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        bound_m: Optional[torch.Tensor] = None,
        do_send: bool = True,
        do_recv: bool = True,
    ) -> None:
        handle = self._ensure_handle(dp_x=dp_x, dp_x_scale=dp_x_scale)
        out_expert_x = _flatten_expert_tensor(out_expert_x, "out_expert_x")
        if out_expert_x_scale is not None:
            out_expert_x_scale = _flatten_expert_tensor(out_expert_x_scale,
                                                        "out_expert_x_scale")
        if indices.dtype != torch.uint32:
            indices = indices.to(torch.uint32)
        if weights is None:
            weights = self._weights_like(indices, 1.0)
        elif weights.dtype != torch.float32:
            weights = weights.to(torch.float32)

        if dp_x.size(0) == 0:
            dp_x = torch.zeros((1, self.hidden_dim),
                               dtype=dp_x.dtype,
                               device=dp_x.device)
            dp_x_scale = self._dummy_scale(dp_x_scale)
            indices = self._dummy_indices(indices)
            weights = self._weights_like(indices, 0.0)
        else:
            weights = self._mask_weights_for_lane(indices, weights)

        handle.dispatch(
            out_expert_num_tokens=out_expert_num_tokens,
            out_expert_x=out_expert_x,
            out_expert_x_scale=out_expert_x_scale,
            dp_x=dp_x,
            dp_x_scale=dp_x_scale,
            indices=indices,
            weights=weights,
            bound_m=_as_int32_bound(bound_m),
            do_send=do_send,
            do_recv=do_recv,
        )
        if do_recv:
            _debug_sync_and_check_counts(
                label="dispatch_recv",
                out_expert_num_tokens=out_expert_num_tokens,
                flat_expert_x=out_expert_x,
            )

    def combine(
        self,
        out_tokens: torch.Tensor,
        indices: torch.Tensor,
        weights: torch.Tensor,
        expert_y: torch.Tensor,
        bound_m: Optional[torch.Tensor],
        do_send: bool = True,
        do_recv: bool = True,
    ) -> None:
        if self._handle is None:
            raise RuntimeError("pplx-garden combine called before dispatch")

        if indices.dtype != torch.uint32:
            indices = indices.to(torch.uint32)
        if weights.dtype != torch.float32:
            weights = weights.to(torch.float32)

        expert_y = _flatten_expert_tensor(expert_y, "expert_y")
        real_out_tokens = out_tokens
        if real_out_tokens.size(0) == 0:
            out_tokens = torch.empty((1, self.hidden_dim),
                                     dtype=real_out_tokens.dtype,
                                     device=real_out_tokens.device)
            indices = self._dummy_indices(indices)
            weights = self._weights_like(indices, 0.0)
        else:
            weights = self._mask_weights_for_lane(indices, weights)

        self._handle.combine(
            out_tokens=out_tokens,
            indices=indices,
            weights=weights,
            expert_y=expert_y,
            bound_m=_as_int32_bound(bound_m),
            do_send=do_send,
            do_recv=do_recv,
        )
        if (do_recv and self.compat_mode == "mask" and self.tp_size > 1
                and real_out_tokens.size(0) > 0):
            if self.node_group is None:
                raise RuntimeError(
                    "pplx-garden mask compatibility mode lost node_group "
                    "before final all-reduce")
            self.node_group.all_reduce(real_out_tokens)
        if do_recv and _env_bool("PPLX_GARDEN_DEBUG_SYNC"):
            torch.cuda.synchronize(real_out_tokens.device)

    def destroy(self) -> None:
        if self._handle is not None:
            self._handle.destroy()
            self._handle = None
        self._weights_cache.clear()
