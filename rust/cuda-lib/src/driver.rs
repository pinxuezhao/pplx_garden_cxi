use std::{
    ffi::{CStr, c_char, c_void},
    ptr::{NonNull, null},
};

use cuda_sys::{
    CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TARGET_CURRENT_CTX,
    CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_OWNER, CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD,
    cuFlushGPUDirectRDMAWrites, cuGetErrorString,
    cuMemGetHandleForAddressRange,
};

type Result<T> = std::result::Result<T, CudaDriverError>;

#[derive(Clone, Debug)]
pub struct CudaDriverError {
    pub code: u32,
    pub context: &'static str,
}

impl CudaDriverError {
    pub fn new(code: u32, context: &'static str) -> Self {
        Self { code, context }
    }
}

impl std::fmt::Display for CudaDriverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut errstr: *const c_char = null();
        unsafe { cuGetErrorString(self.code, &mut errstr) };

        write!(
            f,
            "CudaDriverError: code {} ({:?}), context: {}",
            self.code,
            unsafe { CStr::from_ptr(errstr) },
            self.context
        )
    }
}

impl std::error::Error for CudaDriverError {}

pub fn cu_get_dma_buf_fd(ptr: NonNull<c_void>, len: usize) -> Result<i32> {
    let mut dmabuf_fd: i32 = -1;
    let ret = unsafe {
        cuMemGetHandleForAddressRange(
            &raw mut dmabuf_fd as *mut c_void,
            ptr.as_ptr() as u64,
            len,
            CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD,
            0,
        )
    };
    match ret {
        0 => Ok(dmabuf_fd),
        _ => Err(CudaDriverError::new(ret, "cuMemGetHandleForAddressRange")),
    }
}

pub fn cu_flush_gpu_direct_rdma_writes_to_owner() -> Result<()> {
    let ret = unsafe {
        cuFlushGPUDirectRDMAWrites(
            CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TARGET_CURRENT_CTX,
            CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_OWNER,
        )
    };
    match ret {
        0 => Ok(()),
        _ => Err(CudaDriverError::new(ret, "cuFlushGPUDirectRDMAWrites")),
    }
}
