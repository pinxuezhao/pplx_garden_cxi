#export FI_LOG_LEVEL=debug &&

srun --ntasks-per-node=1 --export=ALL --mpi=pmix -ul \
     --environment=/capstor/scratch/cscs/pzhao/DOCKER/pplx_garden.toml \
     --container-workdir=/capstor/scratch/cscs/pzhao/PPLX_GARDEN/pplx-garden/tests/fabric_lib \
     --output=/capstor/scratch/cscs/pzhao/PPLX_GARDEN/pplx-garden/tests/fabric_lib/log_transfer_engine \
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

        python test_transfer_engine.py

     '
#        export FI_LOG_LEVEL=debug &&
