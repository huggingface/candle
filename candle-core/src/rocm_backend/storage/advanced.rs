//! Advanced operations (convolutions, pooling, matmul, indexing)

use crate::rocm_backend::{miopen, rocblas};
use crate::Result;
use super::RocmStorage;

impl RocmStorage {
    pub(super) fn conv2d_impl(
        &self,
        inp_l: &crate::Layout,
        kernel: &Self,
        kernel_l: &crate::Layout,
        params: &crate::conv::ParamsConv2D,
    ) -> Result<Self> {
        miopen::conv2d(self, inp_l, kernel, kernel_l, params)
    }

    pub(super) fn avg_pool2d_impl(
        &self,
        layout: &crate::Layout,
        k: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Self> {
        self.pool2d(
            layout,
            k,
            stride,
            rocm_rs::miopen::ffi::miopenPoolingMode_t_miopenPoolingAverage,
        )
    }

    pub(super) fn max_pool2d_impl(
        &self,
        layout: &crate::Layout,
        k: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Self> {
        self.pool2d(
            layout,
            k,
            stride,
            rocm_rs::miopen::ffi::miopenPoolingMode_t_miopenPoolingMax,
        )
    }

    pub(super) fn matmul_impl(
        &self,
        rhs: &Self,
        (b, m, n, k): (usize, usize, usize, usize),
        lhs_l: &crate::Layout,
        rhs_l: &crate::Layout,
    ) -> Result<Self> {
        rocblas::matmul(self, rhs, (b, m, n, k), lhs_l, rhs_l)
    }
}
