use crate::{
    lazy::{custom::LazyCustomOp, LazyStorage},
    op::{CmpOp, ReduceOp},
    DType, Layout,
};

pub trait LazyOp {
    fn srcs(&self) -> Vec<&LazyStorage>;
}

#[derive(Debug, Clone, PartialEq)]
pub struct Affine {
    pub src: LazyStorage,
    pub mul: f64,
    pub add: f64,
}

impl Affine {
    pub fn new(src: LazyStorage, mul: f64, add: f64) -> Affine {
        Affine { src, mul, add }
    }
}

impl LazyOp for Affine {
    fn srcs(&self) -> Vec<&LazyStorage> {
        vec![&self.src]
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Powf {
    src: LazyStorage,
    exp: f64,
}

impl Powf {
    pub fn new(src: LazyStorage, exp: f64) -> Powf {
        Powf { src, exp }
    }
}

impl LazyOp for Powf {
    fn srcs(&self) -> Vec<&LazyStorage> {
        vec![&self.src]
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Elu {
    src: LazyStorage,
    alpha: f64,
}

impl Elu {
    pub fn new(src: LazyStorage, alpha: f64) -> Elu {
        Elu { src, alpha }
    }
}

impl LazyOp for Elu {
    fn srcs(&self) -> Vec<&LazyStorage> {
        vec![&self.src]
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Reduce {
    src: LazyStorage,
    reduce_op: ReduceOp,
    dims: Vec<usize>,
    reduction_layout: Layout,
    count: usize,
}

impl Reduce {
    pub fn new(
        src: LazyStorage,
        reduce_op: ReduceOp,
        dims: Vec<usize>,
        reduction_layout: Layout,
        count: usize,
    ) -> Self {
        Self {
            src,
            reduce_op,
            dims,
            reduction_layout,
            count,
        }
    }

    pub fn src(&self) -> &LazyStorage {
        &self.src
    }

    pub fn op(&self) -> &ReduceOp {
        &self.reduce_op
    }

    pub fn reduction_layout(&self) -> &Layout {
        &self.reduction_layout
    }

    pub fn count(&self) -> usize {
        self.count
    }
}

impl LazyOp for Reduce {
    fn srcs(&self) -> Vec<&LazyStorage> {
        vec![&self.src]
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Cmp {
    lhs: LazyStorage,
    rhs: LazyStorage,
    cmp_op: CmpOp,
}

impl Cmp {
    pub fn new(lhs: LazyStorage, rhs: LazyStorage, cmp_op: CmpOp) -> Self {
        Self { lhs, rhs, cmp_op }
    }
}

impl LazyOp for Cmp {
    fn srcs(&self) -> Vec<&LazyStorage> {
        vec![&self.lhs, &self.rhs]
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToDType {
    src: LazyStorage,
    pub dtype: DType,
}

impl ToDType {
    pub fn new(src: LazyStorage, dtype: DType) -> Self {
        Self { src, dtype }
    }
}

impl LazyOp for ToDType {
    fn srcs(&self) -> Vec<&LazyStorage> {
        vec![&self.src]
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Unary {
    src: LazyStorage,
    op: &'static str,
}

impl Unary {
    pub fn new(src: LazyStorage, op: &'static str) -> Self {
        Self { src, op }
    }

    pub fn op(&self) -> &'static str {
        self.op
    }
}

impl LazyOp for Unary {
    fn srcs(&self) -> Vec<&LazyStorage> {
        vec![&self.src]
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Binary {
    lhs: LazyStorage,
    rhs: LazyStorage,
    op: &'static str,
}

impl Binary {
    pub fn new(lhs: LazyStorage, rhs: LazyStorage, op: &'static str) -> Self {
        Self { lhs, rhs, op }
    }

    pub fn op(&self) -> &'static str {
        self.op
    }

    pub fn lhs_l(&self) -> &Layout {
        self.lhs.layout()
    }

    pub fn rhs_l(&self) -> &Layout {
        self.rhs.layout()
    }
}

impl LazyOp for Binary {
    fn srcs(&self) -> Vec<&LazyStorage> {
        vec![&self.lhs, &self.rhs]
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct WhereCond {
    src: LazyStorage,
    t: LazyStorage,
    f: LazyStorage,
}
impl WhereCond {
    pub fn new(src: LazyStorage, t: LazyStorage, f: LazyStorage) -> Self {
        Self { src, t, f }
    }

    pub fn src_l(&self) -> &Layout {
        self.src.layout()
    }

    pub fn t_l(&self) -> &Layout {
        self.t.layout()
    }

    pub fn f_l(&self) -> &Layout {
        self.f.layout()
    }
}

impl LazyOp for WhereCond {
    fn srcs(&self) -> Vec<&LazyStorage> {
        vec![&self.src, &self.t, &self.f]
    }
}

/*
#[derive(Debug, Clone, PartialEq)]
pub struct Conv1D {
    src: LazyStorage,
    params: crate::conv::ParamsConv1D,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ConvTranspose1D {
    src: LazyStorage,
    params: crate::conv::ParamsConvTranspose1D,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Conv2D {
    lhs: LazyStorage,
    rhs: LazyStorage,
    params: crate::conv::ParamsConv2D,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ConvTranspose2D {
    lhs: LazyStorage,
    rhs: LazyStorage,
    params: crate::conv::ParamsConvTranspose2D,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AvgPool2D {
    lhs: LazyStorage,
    rhs: LazyStorage,
    kernel_size: (usize, usize),
    stride: (usize, usize),
}

#[derive(Debug, Clone, PartialEq)]
pub struct MaxPool2D {
    lhs: LazyStorage,
    rhs: LazyStorage,
    kernel_size: (usize, usize),
    stride: (usize, usize),
}

#[derive(Debug, Clone, PartialEq)]
pub struct UpsampleNearest1D {
    src: LazyStorage,
    size: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Gather {
    ids: LazyStorage,
    ids_l: Layout,
    dim: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ScatterSet {
    src: LazyStorage,
    src_l: Layout,
    dim: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ScatterAddSet {
    src: LazyStorage,
    src_l: Layout,
    dim: usize,
}
 */

#[derive(Debug, Clone, PartialEq)]
pub struct IndexSelect {
    src: LazyStorage,
    ids: LazyStorage,
    dim: usize,
}
impl IndexSelect {
    pub fn new(src: LazyStorage, ids: LazyStorage, dim: usize) -> Self {
        Self { src, ids, dim }
    }

    pub fn src_l(&self) -> &Layout {
        self.src.layout()
    }

    pub fn ids_l(&self) -> &Layout {
        self.ids.layout()
    }

    pub fn dim(&self) -> usize {
        self.dim
    }
}

impl LazyOp for IndexSelect {
    fn srcs(&self) -> Vec<&LazyStorage> {
        vec![&self.src, &self.ids]
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Matmul {
    lhs: LazyStorage,
    rhs: LazyStorage,
    b: usize,
    m: usize,
    n: usize,
    k: usize,
}
impl Matmul {
    pub fn new(
        lhs: LazyStorage,
        rhs: LazyStorage,
        (b, m, n, k): (usize, usize, usize, usize),
    ) -> Self {
        Self {
            lhs,
            rhs,
            b,
            m,
            n,
            k,
        }
    }

    pub fn lhs_l(&self) -> &Layout {
        self.lhs.layout()
    }

    pub fn rhs_l(&self) -> &Layout {
        self.rhs.layout()
    }

    pub fn dims(&self) -> (usize, usize, usize, usize) {
        (self.b, self.m, self.n, self.k)
    }
}

impl LazyOp for Matmul {
    fn srcs(&self) -> Vec<&LazyStorage> {
        vec![&self.lhs, &self.rhs]
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SingleCopyStridedSrc {
    pub src: LazyStorage,
    pub dst_offset: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CopyStridedSrc {
    copies: Vec<SingleCopyStridedSrc>,
}

impl CopyStridedSrc {
    pub fn new(src: LazyStorage, dst_offset: usize) -> Self {
        Self {
            copies: vec![SingleCopyStridedSrc { src, dst_offset }],
        }
    }

    pub fn add(&mut self, src: LazyStorage, dst_offset: usize) {
        self.copies.push(SingleCopyStridedSrc { src, dst_offset });
    }

    pub fn copies(&self) -> &[SingleCopyStridedSrc] {
        &self.copies
    }
}

impl LazyOp for CopyStridedSrc {
    fn srcs(&self) -> Vec<&LazyStorage> {
        self.copies.iter().map(|c| &c.src).collect()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SingleCopy2D {
    src: LazyStorage,
    d1: usize,
    d2: usize,
    src_s: usize,
    dst_s: usize,
    src_o: usize,
    dst_o: usize,
}

impl SingleCopy2D {
    pub fn new(
        src: LazyStorage,
        d1: usize,
        d2: usize,
        src_s: usize,
        dst_s: usize,
        src_o: usize,
        dst_o: usize,
    ) -> Self {
        Self {
            src,
            d1,
            d2,
            src_s,
            dst_s,
            src_o,
            dst_o,
        }
    }

    pub fn d1(&self) -> usize {
        self.d1
    }

    pub fn d2(&self) -> usize {
        self.d2
    }

    pub fn src_s(&self) -> usize {
        self.src_s
    }

    pub fn dst_s(&self) -> usize {
        self.dst_s
    }

    pub fn src_o(&self) -> usize {
        self.src_o
    }

    pub fn dst_o(&self) -> usize {
        self.dst_o
    }

    pub(crate) fn src(&self) -> &LazyStorage {
        &self.src
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Copy2D {
    copies: Vec<SingleCopy2D>,
}

impl Copy2D {
    pub fn new(copies: Vec<SingleCopy2D>) -> Self {
        Self { copies }
    }
    pub fn add(&mut self, copy2d: SingleCopy2D) {
        self.copies.push(copy2d);
    }

    pub fn copies(&self) -> &[SingleCopy2D] {
        &self.copies
    }
}

impl LazyOp for Copy2D {
    fn srcs(&self) -> Vec<&LazyStorage> {
        self.copies.iter().map(|c| &c.src).collect()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Sink {
    srcs: Vec<LazyStorage>,
}

impl Sink {
    pub fn new(srcs: Vec<LazyStorage>) -> Self {
        Self { srcs }
    }

    pub fn add(&mut self, src: LazyStorage) {
        self.srcs.push(src);
    }
}

impl LazyOp for Sink {
    fn srcs(&self) -> Vec<&LazyStorage> {
        self.srcs.iter().collect()
    }
}

#[derive(Debug, Clone)]
pub struct CustomOpContainer {
    args: Vec<(LazyStorage, Layout)>,
    op: Box<dyn LazyCustomOp>,
}

impl PartialEq for CustomOpContainer {
    fn eq(&self, other: &Self) -> bool {
        self.args == other.args && self.op.name() == other.op.name()
    }
}

impl LazyOp for CustomOpContainer {
    fn srcs(&self) -> Vec<&LazyStorage> {
        self.args.iter().map(|(b, _)| b).collect()
    }
}

impl CustomOpContainer {
    pub fn new(args: Vec<(LazyStorage, Layout)>, op: Box<dyn LazyCustomOp>) -> CustomOpContainer {
        CustomOpContainer { args, op }
    }

    pub fn args(&self) -> &[(LazyStorage, Layout)] {
        &self.args
    }

    pub fn op(&self) -> &dyn LazyCustomOp {
        &*self.op
    }

    pub fn name(&self) -> &'static str {
        self.op.name()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Output {
    src: LazyStorage,
}

impl Output {
    pub fn new(src: LazyStorage) -> Self {
        Self { src }
    }
}

impl LazyOp for Output {
    fn srcs(&self) -> Vec<&LazyStorage> {
        vec![&self.src]
    }
}
