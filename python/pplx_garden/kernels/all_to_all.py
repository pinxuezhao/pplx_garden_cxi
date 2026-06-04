from typing import Optional, Protocol

import torch


class AllToAllKernel(Protocol):
    def dispatch(
        self,
        out_expert_num_tokens: torch.Tensor,
        out_expert_x: torch.Tensor,
        out_expert_x_scale: Optional[torch.Tensor],
        dp_x: torch.Tensor,
        dp_x_scale: Optional[torch.Tensor],
        indices: torch.Tensor,
        weights: torch.Tensor,
        bound_m: Optional[torch.Tensor] = None,
        do_send: bool = True,
        do_recv: bool = True,
        out_recv_topk_idx: Optional[torch.Tensor] = None,
        out_recv_topk_weights: Optional[torch.Tensor] = None,
        out_recv_src_token_idx: Optional[torch.Tensor] = None,
    ) -> None: ...

    def combine(
        self,
        out_tokens: torch.Tensor,
        indices: torch.Tensor,
        weights: torch.Tensor,
        expert_y: torch.Tensor,
        recv_src_token_idx: Optional[torch.Tensor] = None,
        recv_topk_weights: Optional[torch.Tensor] = None,
        bound_m: Optional[torch.Tensor] = None,
        do_send: bool = True,
        do_recv: bool = True,
        accumulate: bool = False,
    ) -> None: ...

    def destroy(self) -> None: ...
