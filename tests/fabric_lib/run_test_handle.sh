export NCCL_NET="AWS Libfabric" 
export NCCL_NET_PLUGIN="ofi" 
export MASTER_IP="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)" &&
export NODE_RANK=$SLURM_NODEID 
export NUM_NODES=$SLURM_NNODES 
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH 
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH 
export LD_LIBRARY_PATH=/root/libfabric_writedata/install/lib:$LD_LIBRARY_PATH 
export RUST_BACKTRACE=full 
export FI_CXI_ENABLE_WRITEDATA=1      
#export FI_LOG_LEVEL=debug 

python test_handle.py
