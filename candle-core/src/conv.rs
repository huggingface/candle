#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ParamsConv1D {
    pub(crate) b_size: Option<usize>,
    // Maybe we should have a version without l_in as this bit depends on the input and not only on
    // the weights.
    pub(crate) l_in: usize,
    pub(crate) c_out: usize,
    pub(crate) c_in: usize,
    pub(crate) k_size: usize,
    pub(crate) padding: usize,
    pub(crate) stride: usize,
}

impl ParamsConv1D {
    pub(crate) fn l_out(&self) -> usize {
        let dilation = 1;
        (self.l_in + 2 * self.padding - dilation * (self.k_size - 1) - 1) / self.stride + 1
    }

    pub(crate) fn out_dims(&self) -> Vec<usize> {
        let l_out = self.l_out();
        match self.b_size {
            None => vec![self.c_out, l_out],
            Some(n) => vec![n, self.c_out, l_out],
        }
    }
}
