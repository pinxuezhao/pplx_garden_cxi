#!/usr/bin/env python3
"""Minimal pplx-garden dispatch correctness and latency benchmark.

Run inside a 2-node KI Slurm allocation:
  bash scripts_vllm_pplx_garden/run_minimal_pplx_garden_dispatch_2node.sh --profile qwen3-30b

The benchmark forces pplx-garden's native all-to-all into dp_size=1 semantics:
every global rank owns an independent token shard, matching the CLIC minimal
sequence-parallel-style microbench.
"""

import argparse
import sys

from pplx_garden_microbench_common import (
    add_common_args,
    allgather_dicts,
    allocate_dispatch_outputs,
    assert_independent_routing,
    benchmark_dispatch_cuda,
    combine_call,
    create_pplx_context,
    dispatch_call,
    expected_local_counts,
    make_inputs,
    print_run_header,
    resolve_config,
    setup_worker_process,
    write_json,
)


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    return parser.parse_args(argv[1:])


def main(argv):
    args = parse_args(argv)
    (
        torch,
        dist,
        process_group,
        global_group,
        node_group,
        rank,
        world_size,
        local_rank,
    ) = setup_worker_process(__file__, args)

    cfg = resolve_config(args)
    if cfg["num_experts"] % world_size != 0:
        raise SystemExit(
            f"num_experts={cfg['num_experts']} must be divisible by world_size={world_size}"
        )

    if rank == 0:
        print_run_header("dispatch", cfg, args, node_group)
        print(
            f"world_size={world_size} local_rank={local_rank} "
            f"LOCAL_WORLD_SIZE={__import__('os').environ.get('LOCAL_WORLD_SIZE')}",
            flush=True,
        )

    ctx = create_pplx_context(torch, cfg, global_group, node_group, world_size)
    try:
        dp_x, indices, weights = make_inputs(torch, cfg, args, rank)
        routing_digest = assert_independent_routing(dist, indices, rank)
        (
            expert_num_tokens,
            expert_x,
            recv_topk_idx,
            recv_topk_weights,
            recv_src_token_idx,
            out_tokens,
        ) = allocate_dispatch_outputs(torch, cfg, world_size)
        expected_counts, remote_recv_routes, remote_send_routes = expected_local_counts(
            dist, indices, cfg
        )

        def run_dispatch() -> None:
            dispatch_call(
                ctx,
                expert_num_tokens,
                expert_x,
                recv_topk_idx,
                recv_topk_weights,
                recv_src_token_idx,
                dp_x,
                indices,
                weights,
            )

        def drain_combine() -> None:
            combine_call(
                ctx,
                out_tokens,
                expert_x,
                recv_src_token_idx,
                recv_topk_weights,
                indices,
                weights,
            )

        run_dispatch()
        torch.cuda.synchronize()
        actual_counts = expert_num_tokens.detach().cpu().tolist()
        count_match = actual_counts == expected_counts
        meta_valid = True
        for expert, count in enumerate(actual_counts):
            if count <= 0:
                continue
            src = recv_src_token_idx[expert, :count]
            w = recv_topk_weights[expert, :count]
            idx = recv_topk_idx[expert, :count]
            meta_valid = bool(
                meta_valid
                and (src >= 0).all().item()
                and (src < cfg["num_tokens"]).all().item()
                and torch.isfinite(w).all().item()
                and (w > 0).all().item()
                and (w <= 1.00001).all().item()
                and (idx == expert).all().item()
            )
            if not meta_valid:
                break

        # pplx-garden progresses dispatch/combine as a paired protocol. Drain
        # the correctness dispatch before the timed dispatch loop.
        drain_combine()
        torch.cuda.synchronize()

        stats = benchmark_dispatch_cuda(
            torch,
            dist,
            run_dispatch,
            drain_combine,
            args.warmup,
            args.iters,
            args.sync_timing,
        )

        bytes_per_token = cfg["hidden"] * 2
        local = {
            "rank": rank,
            "local_rank": local_rank,
            "routing_digest": routing_digest,
            "count_match": count_match,
            "meta_valid": meta_valid,
            "expected_total": sum(expected_counts),
            "actual_total": sum(actual_counts),
            "remote_recv_routes": remote_recv_routes,
            "remote_send_routes": remote_send_routes,
            "remote_recv_bytes": remote_recv_routes * bytes_per_token,
            "remote_send_bytes": remote_send_routes * bytes_per_token,
            **stats,
        }
        gathered = allgather_dicts(dist, local)
        count_match_all = all(x["count_match"] for x in gathered)
        metadata_valid = all(x["meta_valid"] for x in gathered)
        ok = count_match_all and metadata_valid

        if rank == 0:
            median_max = max(x["median_us"] for x in gathered)
            p99_max = max(x["p99_us"] for x in gathered)
            total_remote_recv = sum(x["remote_recv_bytes"] for x in gathered)
            total_remote_send = sum(x["remote_send_bytes"] for x in gathered)
            payload = {
                "kind": "dispatch",
                "backend": "pplx_garden",
                "semantics": "dp_size_1_independent_rank_routing",
                "ok": ok,
                "count_match_all": count_match_all,
                "metadata_valid": metadata_valid,
                "visible_metadata": [
                    "recv_topk_idx",
                    "recv_topk_weights",
                    "recv_src_token_idx",
                ],
                "config": cfg,
                "world_size": world_size,
                "warmup": args.warmup,
                "iters": args.iters,
                "sync_timing": args.sync_timing,
                "median_us_max_rank": median_max,
                "p99_us_max_rank": p99_max,
                "aggregate_remote_recv_bytes": total_remote_recv,
                "aggregate_remote_send_bytes": total_remote_send,
                "aggregate_remote_recv_gbps_at_median": (
                    total_remote_recv / 1e9 / (median_max / 1e6)
                    if median_max > 0
                    else 0.0
                ),
                "ranks": gathered,
            }
            print("PPLX_GARDEN_DISPATCH_OK", int(ok), flush=True)
            print(f"median_us_max_rank={median_max:.2f}", flush=True)
            print(f"p99_us_max_rank={p99_max:.2f}", flush=True)
            print(
                "aggregate_remote_recv_MB={:.2f} aggregate_remote_send_MB={:.2f}".format(
                    total_remote_recv / 1e6, total_remote_send / 1e6
                ),
                flush=True,
            )
            print(
                "aggregate_remote_recv_GBps_at_median={:.2f}".format(
                    payload["aggregate_remote_recv_gbps_at_median"]
                ),
                flush=True,
            )
            write_json(args.json, payload)

        return 0 if ok else 1
    finally:
        dist.barrier()
        ctx.destroy()
        if node_group is not None:
            node_group.destroy()
        global_group.destroy()
        process_group.destroy()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
