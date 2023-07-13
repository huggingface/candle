use crate::{op::Op, Error, Result, Tensor, TensorId};
use std::collections::HashMap;

impl Tensor {
    /// Return all the nodes that lead to this value in a topologically sorted vec, the first
    /// elements having dependencies on the latter ones, e.g. the first element if any is the
    /// argument.
    /// This assumes that the op graph is a DAG.
    fn sorted_nodes(&self) -> Vec<&Tensor> {
        // The vec of sorted nodes is passed as an owned value rather than a mutable reference
        // to get around some lifetime limitations.
        fn walk<'a>(
            node: &'a Tensor,
            nodes: Vec<&'a Tensor>,
            already_seen: &mut HashMap<TensorId, bool>,
        ) -> (bool, Vec<&'a Tensor>) {
            if let Some(&tg) = already_seen.get(&node.id()) {
                return (tg, nodes);
            }
            let mut track_grad = false;
            let mut nodes = if node.is_variable() {
                // Do not call recursively on the "leaf" nodes.
                track_grad = true;
                nodes
            } else if let Some(op) = node.op() {
                match op {
                    Op::WhereCond(t1, t2, t3) => {
                        let (tg, nodes) = walk(t1, nodes, already_seen);
                        track_grad |= tg;
                        let (tg, nodes) = walk(t2, nodes, already_seen);
                        track_grad |= tg;
                        let (tg, nodes) = walk(t3, nodes, already_seen);
                        track_grad |= tg;
                        nodes
                    }
                    Op::Conv1D {
                        arg: lhs,
                        kernel: rhs,
                        ..
                    }
                    | Op::Add(lhs, rhs)
                    | Op::Mul(lhs, rhs)
                    | Op::Sub(lhs, rhs)
                    | Op::Div(lhs, rhs)
                    | Op::Embedding(lhs, rhs)
                    | Op::Matmul(lhs, rhs) => {
                        let (tg, nodes) = walk(lhs, nodes, already_seen);
                        track_grad |= tg;
                        let (tg, nodes) = walk(rhs, nodes, already_seen);
                        track_grad |= tg;
                        nodes
                    }
                    Op::Cat(args, _) => args.iter().fold(nodes, |nodes, arg| {
                        let (tg, nodes) = walk(arg, nodes, already_seen);
                        track_grad |= tg;
                        nodes
                    }),
                    Op::Affine { arg, mul, .. } => {
                        if *mul == 0. {
                            nodes
                        } else {
                            let (tg, nodes) = walk(arg, nodes, already_seen);
                            track_grad |= tg;
                            nodes
                        }
                    }
                    Op::Reshape(node)
                    | Op::Broadcast(node)
                    | Op::Sum(node, _)
                    | Op::ToDType(node)
                    | Op::ToDevice(node)
                    | Op::Transpose(node, _, _)
                    | Op::Narrow(node, _, _, _)
                    | Op::Softmax(node, _)
                    | Op::Sqr(node)
                    | Op::Sqrt(node)
                    | Op::Gelu(node)
                    | Op::Relu(node)
                    | Op::Elu(node, _)
                    | Op::Exp(node)
                    | Op::Log(node)
                    | Op::Sin(node)
                    | Op::Cos(node)
                    | Op::Abs(node)
                    | Op::Neg(node) => {
                        let (tg, nodes) = walk(node, nodes, already_seen);
                        track_grad |= tg;
                        nodes
                    }
                }
            } else {
                nodes
            };
            already_seen.insert(node.id(), track_grad);
            if track_grad {
                nodes.push(node);
            }
            (track_grad, nodes)
        }
        let (_tg, mut nodes) = walk(self, vec![], &mut HashMap::new());
        nodes.reverse();
        nodes
    }

    pub fn backward(&self) -> Result<GradStore> {
        let sorted_nodes = self.sorted_nodes();
        let mut grads = GradStore::new();
        grads.insert(self, self.ones_like()?.contiguous()?);
        for node in sorted_nodes.iter() {
            if node.is_variable() {
                continue;
            }
            let grad = grads.remove(node).unwrap();
            // TODO: We should perform all these operations in place (or at least not track the
            // whole graph). The only drawback would be if we wanted to support grad of grad but
            // this is out of scope.
            if let Some(op) = node.op() {
                match op {
                    Op::Add(lhs, rhs) => {
                        let lhs_sum_grad = grads.or_insert(lhs)?;
                        *lhs_sum_grad = lhs_sum_grad.add(&grad)?;
                        let rhs_sum_grad = grads.or_insert(rhs)?;
                        *rhs_sum_grad = rhs_sum_grad.add(&grad)?;
                    }
                    Op::Sub(lhs, rhs) => {
                        let lhs_sum_grad = grads.or_insert(lhs)?;
                        *lhs_sum_grad = lhs_sum_grad.add(&grad)?;
                        let rhs_sum_grad = grads.or_insert(rhs)?;
                        *rhs_sum_grad = rhs_sum_grad.sub(&grad)?;
                    }
                    Op::Mul(lhs, rhs) => {
                        let lhs_grad = grad.mul(rhs)?;
                        let lhs_sum_grad = grads.or_insert(lhs)?;
                        *lhs_sum_grad = lhs_sum_grad.add(&lhs_grad)?;
                        let rhs_grad = grad.mul(lhs)?;
                        let rhs_sum_grad = grads.or_insert(rhs)?;
                        *rhs_sum_grad = rhs_sum_grad.add(&rhs_grad)?;
                    }
                    Op::Div(lhs, rhs) => {
                        let lhs_grad = grad.div(rhs)?;
                        let lhs_sum_grad = grads.or_insert(lhs)?;
                        *lhs_sum_grad = lhs_sum_grad.add(&lhs_grad)?;
                        let rhs_grad = grad.mul(lhs)?.div(&rhs.sqr()?)?;
                        let rhs_sum_grad = grads.or_insert(rhs)?;
                        *rhs_sum_grad = rhs_sum_grad.add(&rhs_grad)?;
                    }
                    Op::WhereCond(pred, t, f) => {
                        let zeros = grad.zeros_like()?;
                        let t_sum_grad = grads.or_insert(t)?;
                        let t_grad = pred.where_cond(&grad, &zeros)?;
                        *t_sum_grad = t_sum_grad.add(&t_grad)?;
                        let f_sum_grad = grads.or_insert(f)?;
                        let f_grad = pred.where_cond(&zeros, &grad)?;
                        *f_sum_grad = f_sum_grad.add(&f_grad)?;
                    }
                    Op::Conv1D { .. } => return Err(Error::BackwardNotSupported { op: "conv1d" }),
                    Op::Embedding(_lhs, _rhs) => {
                        return Err(Error::BackwardNotSupported { op: "embedding" })
                    }
                    Op::Matmul(lhs, rhs) => {
                        // Skipping checks, the op went ok, we can skip
                        // the matmul size checks for now.

                        let lhs_grad = grad.matmul(&rhs.t()?)?;
                        let lhs_sum_grad = grads.or_insert(lhs)?;
                        *lhs_sum_grad = lhs_sum_grad.add(&lhs_grad)?;

                        let rhs_grad = lhs.t()?.matmul(&grad)?;
                        let rhs_sum_grad = grads.or_insert(rhs)?;
                        *rhs_sum_grad = rhs_sum_grad.add(&rhs_grad)?;
                    }
                    Op::Cat(args, dim) => {
                        let mut start_idx = 0;
                        for arg in args {
                            let len = arg.dims()[*dim];
                            let arg_grad = grad.narrow(*dim, start_idx, len)?;
                            let sum_grad = grads.or_insert(arg)?;
                            *sum_grad = sum_grad.add(&arg_grad)?;
                            start_idx += len;
                        }
                    }
                    Op::Broadcast(arg) => {
                        let arg_dims = arg.dims();
                        let node_dims = node.dims();
                        // The number of dims that have been inserted on the left.
                        let left_dims = node_dims.len() - arg_dims.len();
                        let mut sum_dims: Vec<usize> = (0..left_dims).collect();
                        for (dim, (node_dim, arg_dim)) in node_dims[left_dims..]
                            .iter()
                            .zip(arg_dims.iter())
                            .enumerate()
                        {
                            if node_dim != arg_dim {
                                sum_dims.push(dim + left_dims)
                            }
                        }

                        let arg_grad = grad.sum(sum_dims.as_slice())?;
                        let sum_grad = grads.or_insert(arg)?;
                        *sum_grad = sum_grad.broadcast_add(&arg_grad)?
                    }
                    Op::Sum(arg, _sum_dims) => {
                        let sum_grad = grads.or_insert(arg)?;
                        *sum_grad = sum_grad.broadcast_add(&grad)?
                    }
                    Op::ToDType(arg) => {
                        let sum_grad = grads.or_insert(arg)?;
                        *sum_grad = sum_grad.add(&grad.to_dtype(node.dtype())?)?
                    }
                    Op::Affine { arg, mul, .. } => {
                        let arg_grad = grad.affine(*mul, 0.)?;
                        let sum_grad = grads.or_insert(arg)?;
                        *sum_grad = sum_grad.add(&arg_grad)?
                    }
                    Op::Log(arg) => {
                        let sum_grad = grads.or_insert(arg)?;
                        *sum_grad = sum_grad.add(&(&grad * *node)?)?
                    }
                    Op::Sin(arg) => {
                        let sum_grad = grads.or_insert(arg)?;
                        *sum_grad = sum_grad.add(&(&grad * arg.cos())?)?
                    }
                    Op::Cos(arg) => {
                        let sum_grad = grads.or_insert(arg)?;
                        *sum_grad = sum_grad.sub(&(&grad * arg.sin())?)?
                    }
                    Op::Abs(_args) => return Err(Error::BackwardNotSupported { op: "abs" }),
                    Op::Exp(arg) => {
                        let sum_grad = grads.or_insert(arg)?;
                        *sum_grad = sum_grad.add(&(&grad / arg)?)?
                    }
                    Op::Neg(arg) => {
                        let sum_grad = grads.or_insert(arg)?;
                        *sum_grad = sum_grad.sub(&grad)?
                    }
                    &Op::Narrow(ref arg, dim, start_idx, len) => {
                        let arg_dims = arg.dims();
                        let left_pad = if start_idx == 0 {
                            None
                        } else {
                            let mut dims = arg_dims.to_vec();
                            dims[dim] = start_idx;
                            Some(Tensor::zeros(dims, grad.dtype(), grad.device())?)
                        };
                        let right_pad = arg_dims[dim] - start_idx - len;
                        let right_pad = if right_pad == 0 {
                            None
                        } else {
                            let mut dims = arg_dims.to_vec();
                            dims[dim] = right_pad;
                            Some(Tensor::zeros(dims, grad.dtype(), grad.device())?)
                        };
                        let arg_grad = match (left_pad, right_pad) {
                            (None, None) => grad,
                            (Some(l), None) => Tensor::cat(&[&l, &grad], dim)?,
                            (None, Some(r)) => Tensor::cat(&[&grad, &r], dim)?,
                            (Some(l), Some(r)) => Tensor::cat(&[&l, &grad, &r], dim)?,
                        };
                        let sum_grad = grads.or_insert(arg)?;
                        *sum_grad = sum_grad.add(&arg_grad)?
                    }
                    Op::Softmax(_arg, _) => {
                        return Err(Error::BackwardNotSupported { op: "softmax" })
                    }
                    Op::Reshape(arg) => {
                        let arg_grad = grad.reshape(arg.dims())?;
                        let sum_grad = grads.or_insert(arg)?;
                        *sum_grad = sum_grad.add(&arg_grad)?
                    }
                    Op::Gelu(_) => return Err(Error::BackwardNotSupported { op: "gelu" }),
                    Op::Relu(_) => return Err(Error::BackwardNotSupported { op: "relu" }),
                    Op::Elu(..) => return Err(Error::BackwardNotSupported { op: "elu" }),
                    Op::Sqr(arg) => {
                        let arg_grad = arg.mul(&grad)?.affine(2., 0.)?;
                        let sum_grad = grads.or_insert(arg)?;
                        *sum_grad = sum_grad.add(&arg_grad)?
                    }
                    Op::Sqrt(arg) => {
                        let arg_grad = grad.div(arg)?.affine(0.5, 0.)?;
                        let sum_grad = grads.or_insert(arg)?;
                        *sum_grad = sum_grad.add(&arg_grad)?
                    }
                    Op::ToDevice(arg) => {
                        let sum_grad = grads.or_insert(arg)?;
                        let arg_grad = grad.to_device(sum_grad.device())?;
                        *sum_grad = sum_grad.add(&arg_grad)?
                    }
                    Op::Transpose(arg, dim1, dim2) => {
                        let arg_grad = grad.transpose(*dim1, *dim2)?;
                        let sum_grad = grads.or_insert(arg)?;
                        *sum_grad = sum_grad.add(&arg_grad)?
                    }
                };
            }
        }
        Ok(grads)
    }
}

pub struct GradStore(HashMap<TensorId, Tensor>);

impl GradStore {
    fn new() -> Self {
        GradStore(HashMap::new())
    }

    pub fn get_id(&self, id: TensorId) -> Option<&Tensor> {
        self.0.get(&id)
    }

    pub fn get(&self, tensor: &Tensor) -> Option<&Tensor> {
        self.0.get(&tensor.id())
    }

    pub fn remove(&mut self, tensor: &Tensor) -> Option<Tensor> {
        self.0.remove(&tensor.id())
    }

    pub fn insert(&mut self, tensor: &Tensor, grad: Tensor) -> Option<Tensor> {
        self.0.insert(tensor.id(), grad)
    }

    fn or_insert(&mut self, tensor: &Tensor) -> Result<&mut Tensor> {
        use std::collections::hash_map::Entry;
        let grad = match self.0.entry(tensor.id()) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => {
                let grad = tensor.zeros_like()?;
                entry.insert(grad)
            }
        };
        Ok(grad)
    }
}
