#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ParamsConv1D {
    pub(crate) b_size: Option<usize>,
    pub(crate) c_out: usize,
    pub(crate) c_in: usize,
    pub(crate) k_size: usize,
    pub(crate) padding: usize,
    pub(crate) stride: usize,
}

impl ParamsConv1D {
    pub(crate) fn l_out(&self, l_in: usize) -> usize {
        let dilation = 1;
        (l_in + 2 * self.padding - dilation * (self.k_size - 1) - 1) / self.stride + 1
    }

    pub(crate) fn out_dims(&self, l_in: usize) -> Vec<usize> {
        let l_out = self.l_out(l_in);
        match self.b_size {
            None => vec![self.c_out, l_out],
            Some(n) => vec![n, self.c_out, l_out],
        }
    }
}
