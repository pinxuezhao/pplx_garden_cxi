export LIBFABRIC_HOME=/root/libfabric_writedata/install
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH
cargo build --release --bin fabric-debug
