#export FI_LOG_LEVEL=debug &&

srun --ntasks-per-node=1 --export=ALL --mpi=pmix -ul \
     --environment=/capstor/scratch/cscs/pzhao/DOCKER/pplx_garden.toml \
     --container-workdir=/capstor/scratch/cscs/pzhao/PPLX_GARDEN/pplx-garden \
     --output=/capstor/scratch/cscs/pzhao/PPLX_GARDEN/pplx-garden/log_benchmark_a2a \
     --partition=debug \
     bash -lc '
        set -x

        source /capstor/scratch/cscs/pzhao/PPLX_GARDEN/venv_pplx/bin/activate &&
        export NCCL_NET="AWS Libfabric" &&
        export NCCL_NET_PLUGIN="ofi" &&
        export MASTER_IP="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)" &&
        export NODE_RANK=$SLURM_NODEID &&
        export NUM_NODES=$SLURM_NNODES &&
        export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH &&
        export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH &&
        export LD_LIBRARY_PATH=/root/libfabric_writedata/install/lib:$LD_LIBRARY_PATH &&
        export RUST_BACKTRACE=full &&
        export FI_CXI_ENABLE_WRITEDATA=1 &&



        export PATH="/root/.cargo/bin:$PATH"
        export CARGO_HOME="/root/.cargo"
        export RUSTUP_HOME="/root/.rustup"


        python3 -m benchmarks.bench_all_to_all \
            --world-size $((NUM_NODES * 4)) --nets-per-gpu 1 --init-method=tcp://$MASTER_IP:29500 \
            --node-rank=$NODE_RANK \
            --dp-size 1 \
            --max-num-tokens 128 \
            --max-private-tokens 128 \
            --num-experts 256  \
            --hidden-dim 7168 \
            --hidden-dim-scale 56 \
            --num-experts-per-token 8 \
            --in-dtype=float16 \
            --out-dtype=float16 \
            --scale-dtype=float32 \
            --nvlink 4 \
            --num-warmup 20 \
            --num-repeats 30 \

     '
