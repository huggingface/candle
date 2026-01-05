use crate::onnx::attribute_proto::AttributeType;
use crate::onnx::tensor_proto::DataType;
use crate::onnx::{self, GraphProto};
use candle::Module;
use candle::{bail, DType, Device, IndexOp, Result, Tensor};
use candle_nn::activation::PReLU;
use std::collections::{HashMap, HashSet};

pub type Value = Tensor;

pub fn dtype(dt: DataType) -> Option<DType> {
    match dt {
        DataType::Uint8 => Some(DType::U8),
        DataType::Uint32 => Some(DType::U32),
        DataType::Int32 => Some(DType::I64), // Map Int32 to I64 (candle doesn't have I32)
        DataType::Int64 => Some(DType::I64),
        DataType::Float16 => Some(DType::F16),
        DataType::Float => Some(DType::F32),
        DataType::Double => Some(DType::F64),
        DataType::Bool => Some(DType::U8),
        _ => None,
    }
}

trait Attr {
    const TYPE: AttributeType;
    fn get(attr: &onnx::AttributeProto) -> Result<&Self>;
}

trait AttrOwned: Sized {
    const TYPE: AttributeType;
    fn get(attr: &onnx::AttributeProto) -> Result<Self>;
}

impl Attr for i64 {
    const TYPE: AttributeType = AttributeType::Int;
    fn get(attr: &onnx::AttributeProto) -> Result<&Self> {
        Ok(&attr.i)
    }
}

impl Attr for f32 {
    const TYPE: AttributeType = AttributeType::Float;
    fn get(attr: &onnx::AttributeProto) -> Result<&Self> {
        Ok(&attr.f)
    }
}

impl Attr for [i64] {
    const TYPE: AttributeType = AttributeType::Ints;
    fn get(attr: &onnx::AttributeProto) -> Result<&Self> {
        Ok(attr.ints.as_slice())
    }
}

impl Attr for str {
    const TYPE: AttributeType = AttributeType::String;
    fn get(attr: &onnx::AttributeProto) -> Result<&Self> {
        std::str::from_utf8(&attr.s).map_err(candle::Error::wrap)
    }
}

impl Attr for GraphProto {
    const TYPE: AttributeType = AttributeType::Graph;
    fn get(attr: &onnx::AttributeProto) -> Result<&Self> {
        attr.g
            .as_ref()
            .ok_or_else(|| candle::Error::Msg("attribute does not contain graph".to_string()))
    }
}

impl AttrOwned for Vec<String> {
    const TYPE: AttributeType = AttributeType::Strings;
    fn get(attr: &onnx::AttributeProto) -> Result<Self> {
        let mut ret = vec![];
        for bytes in attr.strings.iter() {
            let s = String::from_utf8(bytes.clone()).map_err(candle::Error::wrap)?;
            ret.push(s);
        }
        Ok(ret)
    }
}

impl AttrOwned for Tensor {
    const TYPE: AttributeType = AttributeType::Tensor;
    fn get(attr: &onnx::AttributeProto) -> Result<Self> {
        let tensor_proto = match &attr.t {
            Some(value) => value,
            None => bail!(
                "attribute {} was of type TENSOR, but no tensor was found",
                attr.name
            ),
        };

        let data_type = match DataType::try_from(tensor_proto.data_type) {
            Ok(value) => value,
            Err(_) => bail!(
                "attribute {} of type TENSOR was an invalid data_type number {}",
                attr.name,
                tensor_proto.data_type
            ),
        };

        let dtype = match dtype(data_type) {
            Some(value) => value,
            None => bail!(
                "attribute {} of type TENSOR has an unsupported data_type {}",
                attr.name,
                data_type.as_str_name()
            ),
        };

        let mut dims = Vec::with_capacity(tensor_proto.dims.len());
        for dim in &tensor_proto.dims {
            if dim < &0 {
                bail!(
                    "attribute {} of type TENSOR has a negative dimension, which is unsupported",
                    attr.name
                )
            }
            dims.push(*dim as usize)
        }

        Tensor::from_raw_buffer(&tensor_proto.raw_data, dtype, &dims, &Device::Cpu)
    }
}

fn get_attr_<'a>(node: &'a onnx::NodeProto, name: &str) -> Result<&'a onnx::AttributeProto> {
    match node.attribute.iter().find(|attr| attr.name == name) {
        None => {
            bail!(
                "cannot find the '{name}' attribute in '{}' for {}",
                node.op_type,
                node.name
            )
        }
        Some(dt) => Ok(dt),
    }
}

fn get_attr<'a, T: Attr + ?Sized>(node: &'a onnx::NodeProto, name: &str) -> Result<&'a T> {
    let attr = get_attr_(node, name)?;
    if attr.r#type() != T::TYPE {
        bail!(
            "unsupported type {:?} for '{name}' attribute in '{}' for {}",
            attr.r#type,
            node.op_type,
            node.name
        )
    }
    T::get(attr)
}

fn get_attr_opt<'a, T: Attr + ?Sized>(
    node: &'a onnx::NodeProto,
    name: &str,
) -> Result<Option<&'a T>> {
    match node.attribute.iter().find(|attr| attr.name == name) {
        None => Ok(None),
        Some(attr) => {
            if attr.r#type() != T::TYPE {
                bail!(
                    "unsupported type {:?} for '{name}' attribute in '{}' for {}",
                    attr.r#type,
                    node.op_type,
                    node.name
                )
            }
            let val = T::get(attr)?;
            Ok(Some(val))
        }
    }
}

fn get_attr_opt_owned<T: AttrOwned>(node: &onnx::NodeProto, name: &str) -> Result<Option<T>> {
    match node.attribute.iter().find(|attr| attr.name == name) {
        None => Ok(None),
        Some(attr) => {
            if attr.r#type() != T::TYPE {
                bail!(
                    "unsupported type {:?} for '{name}' attribute in '{}' for {}",
                    attr.r#type,
                    node.op_type,
                    node.name
                )
            }
            let val = T::get(attr)?;
            Ok(Some(val))
        }
    }
}

pub fn get_tensor(t: &onnx::TensorProto, name: &str) -> Result<Tensor> {
    let dims: Vec<usize> = t.dims.iter().map(|&x| x as usize).collect();
    match DataType::try_from(t.data_type) {
        Ok(DataType::Int32) => {
            if t.int32_data.is_empty() {
                let len = t.raw_data.len() / 4;
                let data: &[i32] =
                    unsafe { std::slice::from_raw_parts(t.raw_data.as_ptr() as *const i32, len) };
                let data = data.iter().map(|v| *v as i64).collect::<Vec<_>>();
                Tensor::from_vec(data, len, &Device::Cpu)
            } else {
                let data = t.int32_data.iter().map(|v| *v as i64).collect::<Vec<_>>();
                Tensor::from_vec(data, t.int32_data.len(), &Device::Cpu)
            }
        }
        Ok(dt) => match dtype(dt) {
            Some(dt) => {
                if dt == DType::F32 && !t.float_data.is_empty() {
                    Tensor::from_slice(&t.float_data, dims.as_slice(), &Device::Cpu)
                } else if dt == DType::F64 && !t.double_data.is_empty() {
                    Tensor::from_slice(&t.double_data, dims.as_slice(), &Device::Cpu)
                } else if dt == DType::I64 && !t.int64_data.is_empty() {
                    Tensor::from_slice(&t.int64_data, dims.as_slice(), &Device::Cpu)
                } else {
                    Tensor::from_raw_buffer(
                        t.raw_data.as_slice(),
                        dt,
                        dims.as_slice(),
                        &Device::Cpu,
                    )
                }
            }
            None => {
                bail!("unsupported 'value' data-type {dt:?} for {name}")
            }
        },
        Err(_) => {
            bail!("unsupported 'value' data-type {} for {name}", t.data_type,)
        }
    }
}

// This function provides a direct evaluation of the proto.
// Longer-term, we should first convert the proto to an intermediate representation of the compute
// graph so as to make multiple evaluations more efficient.
// An example upside of this would be to remove intermediary values when they are not needed
// anymore.
pub fn simple_eval(
    model: &onnx::ModelProto,
    mut inputs: HashMap<String, Value>,
) -> Result<HashMap<String, Value>> {
    let graph = match &model.graph {
        None => bail!("no graph defined in proto"),
        Some(graph) => graph,
    };
    simple_eval_(graph, &mut inputs)
}

fn simple_eval_(
    graph: &onnx::GraphProto,
    values: &mut HashMap<String, Value>,
) -> Result<HashMap<String, Value>> {
    for t in graph.initializer.iter() {
        let tensor = get_tensor(t, t.name.as_str())?;
        values.insert(t.name.to_string(), tensor);
    }
    for input in graph.input.iter() {
        let input_type = match &input.r#type {
            Some(input_type) => input_type,
            None => continue,
        };
        let input_type = match &input_type.value {
            Some(input_type) => input_type,
            None => continue,
        };
        let tensor_type = match input_type {
            onnx::type_proto::Value::TensorType(tt) => tt,
            _ => continue,
        };

        let tensor = match values.get(&input.name) {
            None => bail!("missing input {}", input.name),
            Some(tensor) => tensor,
        };
        let dt = match DataType::try_from(tensor_type.elem_type) {
            Ok(dt) => match dtype(dt) {
                Some(dt) => dt,
                None => {
                    bail!("unsupported 'value' data-type {dt:?} for {}", input.name)
                }
            },
            type_ => bail!("unsupported input type {type_:?}"),
        };
        match &tensor_type.shape {
            None => continue,
            Some(shape) => {
                if shape.dim.len() != tensor.rank() {
                    bail!(
                        "unexpected rank for {}, got {:?}, expected {:?}",
                        input.name,
                        shape.dim,
                        tensor.shape()
                    )
                }
                for (idx, (d, &dim)) in shape.dim.iter().zip(tensor.dims().iter()).enumerate() {
                    match &d.value {
                        Some(onnx::tensor_shape_proto::dimension::Value::DimValue(v)) => {
                            if *v as usize != dim {
                                bail!(
                                    "unexpected dim {idx} for {}, got {:?}, expected {:?}",
                                    input.name,
                                    shape.dim,
                                    tensor.shape()
                                )
                            }
                        }
                        // We do not check equality constraints for the DimParam dimensions for now.
                        Some(onnx::tensor_shape_proto::dimension::Value::DimParam(_)) | None => (),
                    }
                }
            }
        };
        if dt != tensor.dtype() {
            bail!(
                "unexpected dtype for {}, got {:?}, expected {dt:?}",
                input.name,
                tensor.dtype()
            )
        }
    }
    // The nodes are topologically sorted so we can just process them in order.
    for node in graph.node.iter() {
        let get = |input_name: &str| match values.get(input_name) {
            Some(value) => Ok(value),
            None => bail!("cannot find {input_name} for op '{}'", node.name),
        };
        let get_opt = |i: usize| {
            node.input
                .get(i)
                .filter(|s: &&String| !s.is_empty())
                .map(|s| get(s))
        };

        // TODO: Validate node.input for each operator.
        match node.op_type.as_str() {
            "Add" => {
                let input0 = get(&node.input[0])?;
                let input1 = get(&node.input[1])?;
                let output = input0.broadcast_add(input1)?;
                values.insert(node.output[0].clone(), output);
            }
            "Sub" => {
                let input0 = get(&node.input[0])?;
                let input1 = get(&node.input[1])?;
                let output = input0.broadcast_sub(input1)?;
                values.insert(node.output[0].clone(), output);
            }
            "Mul" => {
                let input0 = get(&node.input[0])?;
                let input1 = get(&node.input[1])?;
                let output = input0.broadcast_mul(input1)?;
                values.insert(node.output[0].clone(), output);
            }
            "Div" => {
                let input0 = get(&node.input[0])?;
                let input1 = get(&node.input[1])?;
                let output = input0.broadcast_div(input1)?;
                values.insert(node.output[0].clone(), output);
            }
            "Pow" => {
                let input0 = get(&node.input[0])?;
                let input1 = get(&node.input[1])?;
                // HACK: current implementation of broadcast_pow cannot handle negative base,
                // so we use powf where we can, which *does* correctly handle negative base.
                if let Ok(exp) = to_scalar_flexible::<f64>(&input1.to_dtype(DType::F64)?) {
                    let output = input0.powf(exp)?;
                    values.insert(node.output[0].clone(), output);
                } else {
                    let output = input0.broadcast_pow(input1)?;
                    values.insert(node.output[0].clone(), output);
                }
            }
            "Exp" => {
                let xs = get(&node.input[0])?;
                let output = xs.exp()?;
                values.insert(node.output[0].clone(), output);
            }
            "Equal" => {
                let input0 = get(&node.input[0])?;
                let input1 = get(&node.input[1])?;
                let output = input0.broadcast_eq(input1)?;
                values.insert(node.output[0].clone(), output);
            }
            "Not" => {
                let xs = get(&node.input[0])?;
                let xs = xs.eq(&xs.zeros_like()?)?;
                values.insert(node.output[0].clone(), xs);
            }
            "MatMul" => {
                let input0 = get(&node.input[0])?;
                let input1 = get(&node.input[1])?;
                let output = input0.broadcast_matmul(input1)?;
                values.insert(node.output[0].clone(), output);
            }
            "Reshape" => {
                let input0 = get(&node.input[0])?;
                let input1 = get(&node.input[1])?.to_vec1::<i64>()?;
                // TODO: Check that there is at most a single -1 or 0, handle other neg values.
                let mut other_than_minus1 = 1usize;
                for &v in input1.iter() {
                    if v != -1 && v != 0 {
                        other_than_minus1 *= v as usize
                    }
                }
                let input1 = input1
                    .iter()
                    .enumerate()
                    .map(|(idx, &v)| match v {
                        -1 => Ok(input0.elem_count() / other_than_minus1),
                        0 => input0.dim(idx),
                        _ => Ok(v as usize),
                    })
                    .collect::<Result<Vec<usize>>>()?;
                let output = input0.reshape(input1)?;
                values.insert(node.output[0].clone(), output);
            }
            "LogSoftmax" => {
                let input = get(&node.input[0])?;
                let output = match get_attr_opt::<i64>(node, "axis")? {
                    None => candle_nn::ops::softmax_last_dim(input)?,
                    Some(&axis) => {
                        let axis = input.normalize_axis(axis)?;
                        candle_nn::ops::log_softmax(input, axis)?
                    }
                };
                values.insert(node.output[0].clone(), output);
            }
            "Softmax" => {
                let input = get(&node.input[0])?;
                let output = match get_attr_opt::<i64>(node, "axis")? {
                    None => candle_nn::ops::softmax_last_dim(input)?,
                    Some(&axis) => {
                        let axis = input.normalize_axis(axis)?;
                        candle_nn::ops::softmax(input, axis)?
                    }
                };
                values.insert(node.output[0].clone(), output);
            }
            "Transpose" => {
                let input = get(&node.input[0])?;
                let output = match get_attr_opt::<[i64]>(node, "perm")? {
                    None => input.t()?,
                    Some(perm) => {
                        let perm = perm.iter().map(|&v| v as usize).collect::<Vec<_>>();
                        input.permute(perm)?.contiguous()?
                    }
                };
                values.insert(node.output[0].clone(), output);
            }
            "Dropout" => {
                let input = get(&node.input[0])?;
                // Do not apply dropout at the moment, consider that we're only doing inference.
                values.insert(node.output[0].clone(), input.clone());
            }
            "MaxPool" => {
                // https://github.com/onnx/onnx/blob/main/docs/Operators.md#MaxPool
                let dilations = get_attr_opt::<[i64]>(node, "dilations")?;
                let kernel_shape = get_attr::<[i64]>(node, "kernel_shape")?;
                let pads = get_attr_opt::<[i64]>(node, "pads")?;
                let strides = get_attr_opt::<[i64]>(node, "strides")?;
                let auto_pad = get_attr_opt::<str>(node, "auto_pad")?;
                match auto_pad {
                    None | Some("NOTSET") => (),
                    Some(s) => bail!("unsupported auto_pad {s}"),
                };
                if let Some(d) = dilations {
                    if d.iter().any(|&v| v != 1) {
                        bail!("MaxPool with dilation != 1, {dilations:?}")
                    }
                }
                if let Some(d) = pads {
                    if d.iter().any(|&v| v != 0) {
                        bail!("MaxPool with pads != 0, {pads:?}")
                    }
                }
                let xs = get(&node.input[0])?;
                let (k1, k2) = match kernel_shape {
                    [k1, k2] => (*k1 as usize, *k2 as usize),
                    _ => bail!("only 2d MaxPool is supported, kernel shape {kernel_shape:?}"),
                };
                let ys = match strides {
                    None => xs.max_pool2d((k1, k2))?,
                    Some([s1, s2]) => {
                        xs.max_pool2d_with_stride((k1, k2), (*s1 as usize, *s2 as usize))?
                    }
                    Some(strides) => bail!("only 2d MaxPool is supported, strides {strides:?}"),
                };
                values.insert(node.output[0].clone(), ys);
            }
            "AveragePool" => {
                // https://github.com/onnx/onnx/blob/main/docs/Operators.md#AveragePool
                let dilations = get_attr_opt::<[i64]>(node, "dilations")?;
                let kernel_shape = get_attr::<[i64]>(node, "kernel_shape")?;
                let pads = get_attr_opt::<[i64]>(node, "pads")?;
                let strides = get_attr_opt::<[i64]>(node, "strides")?;
                let auto_pad = get_attr_opt::<str>(node, "auto_pad")?;
                let ceil_mode = get_attr_opt::<i64>(node, "ceil_mode")?.copied().unwrap_or(0) == 1;
                match auto_pad {
                    None | Some("NOTSET") => (),
                    Some(s) => bail!("unsupported auto_pad {s}"),
                };
                if let Some(d) = dilations {
                    if d.iter().any(|&v| v != 1) {
                        bail!("AvgPool with dilation != 1, {dilations:?}")
                    }
                }
                // Check pads - only fail if there are non-zero pads
                if let Some(d) = pads {
                    if d.iter().any(|&v| v != 0) {
                        bail!("AvgPool with pads != 0, {pads:?}")
                    }
                }
                let xs = get(&node.input[0])?;
                let ys = match kernel_shape {
                    [k1] => {
                        // 1D average pooling
                        let k = *k1 as usize;
                        let stride = match strides {
                            None => k,
                            Some([s]) => *s as usize,
                            Some(s) => bail!("unexpected strides for 1d AvgPool {s:?}"),
                        };
                        // xs shape: [N, C, L]
                        let (_n, _c, l) = xs.dims3()?;
                        
                        // Handle case where kernel is larger than input
                        if k > l {
                            // Global average pooling over the entire sequence
                            xs.mean(2)?.unsqueeze(2)?
                        } else {
                            // Calculate output length based on ceil_mode
                            let out_len = if ceil_mode {
                                // ceil((l - k) / stride) + 1
                                (l - k + stride - 1) / stride + 1
                            } else {
                                // floor((l - k) / stride) + 1
                                (l - k) / stride + 1
                            };
                            
                            if out_len == 0 {
                                // Return empty tensor with correct shape
                                let mut shape = xs.dims().to_vec();
                                shape[2] = 0;
                                Tensor::zeros(shape, xs.dtype(), xs.device())?
                            } else {
                                // Manual 1D average pooling
                                let mut pools = Vec::with_capacity(out_len);
                                for i in 0..out_len {
                                    let start = i * stride;
                                    // In ceil_mode, the last window might extend beyond input
                                    // We need to handle partial windows
                                    let actual_k = k.min(l - start);
                                    let slice = xs.narrow(2, start, actual_k)?;
                                    let avg = slice.mean(2)?;
                                    pools.push(avg);
                                }
                                Tensor::stack(&pools, 2)?
                            }
                        }
                    }
                    [k1, k2] => {
                        // 2D average pooling
                        let (k1, k2) = (*k1 as usize, *k2 as usize);
                        // Note: 2D pooling with ceil_mode not yet implemented
                        if ceil_mode {
                            bail!("ceil_mode not yet supported for 2D AveragePool")
                        }
                        match strides {
                            None => xs.avg_pool2d((k1, k2))?,
                            Some([s1, s2]) => {
                                xs.avg_pool2d_with_stride((k1, k2), (*s1 as usize, *s2 as usize))?
                            }
                            Some(strides) => bail!("unexpected strides for 2d AvgPool {strides:?}"),
                        }
                    }
                    _ => bail!("only 1d and 2d AvgPool is supported, kernel shape {kernel_shape:?}"),
                };
                values.insert(node.output[0].clone(), ys);
            }
            "BatchNormalization" => {
                let training_mode = get_attr_opt::<i64>(node, "training_mode")?;
                if training_mode.copied().unwrap_or(0) != 0 {
                    bail!("training mode is not supported for BatchNorm")
                }
                let eps = get_attr_opt::<f32>(node, "epsilon")?
                    .copied()
                    .unwrap_or(1e-5);
                let xs = get(&node.input[0])?;
                let weight = get(&node.input[1])?;
                let bias = get(&node.input[2])?;
                let running_mean = get(&node.input[3])?;
                let running_var = get(&node.input[4])?;
                let target_shape: Vec<usize> = xs
                    .dims()
                    .iter()
                    .enumerate()
                    .map(|(idx, v)| if idx == 1 { *v } else { 1 })
                    .collect();
                let target_shape = target_shape.as_slice();
                let xs = xs
                    .broadcast_sub(&running_mean.reshape(target_shape)?)?
                    .broadcast_div(&(running_var.reshape(target_shape)? + eps as f64)?.sqrt()?)?;
                let weight = weight.reshape(target_shape)?;
                let bias = bias.reshape(target_shape)?;
                let xs = xs.broadcast_mul(&weight)?.broadcast_add(&bias)?;
                values.insert(node.output[0].clone(), xs);
            }
            "Squeeze" => {
                let xs = get(&node.input[0])?;
                let mut axes = if node.input.len() <= 1 {
                    // contract all the dimensions with size 1 except the batch dim.
                    xs.dims()
                        .iter()
                        .enumerate()
                        .flat_map(|(idx, &s)| if s == 1 && idx > 0 { Some(idx) } else { None })
                        .collect()
                } else {
                    get(&node.input[1])?
                        .to_vec1::<i64>()?
                        .iter()
                        .map(|&i| xs.normalize_axis(i))
                        .collect::<Result<Vec<_>>>()?
                };
                axes.sort();
                let mut xs = xs.clone();
                for &axis in axes.iter().rev() {
                    xs = xs.squeeze(axis)?
                }
                values.insert(node.output[0].clone(), xs);
            }
            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#ConstantOfShape
            "ConstantOfShape" => {
                let input = get(&node.input[0])?;
                let value = get_attr_opt_owned::<Tensor>(node, "value")?.unwrap_or(Tensor::zeros(
                    (),
                    DType::F32,
                    &Device::Cpu,
                )?);

                let shape_vec: Vec<usize> = input
                    .to_vec1::<i64>()?
                    .iter()
                    .map(|&x| x as usize)
                    .collect();

                let xs = Tensor::ones(shape_vec, value.dtype(), input.device())?
                    .broadcast_mul(&value)?;
                values.insert(node.output[0].clone(), xs);
            }
            "Unsqueeze" => {
                let xs = get(&node.input[0])?;
                let axes = match get_attr_opt::<[i64]>(node, "axes")? {
                    Some(axis) => axis.to_vec(),
                    None => get(&node.input[1])?.to_vec1::<i64>()?,
                };
                let mut axes = axes
                    .iter()
                    .map(|&i| {
                        if i == xs.rank() as i64 {
                            Ok(xs.rank())
                        } else if i < 0 {
                            // normalize_axis doesn't work correctly here
                            // because we actually want normalized with respect
                            // to the final size, not the current (off by one)
                            Ok(xs.rank() - (-i as usize) + 1)
                        } else {
                            xs.normalize_axis(i)
                        }
                    })
                    .collect::<Result<Vec<_>>>()?;
                axes.sort();
                let mut xs = xs.clone();
                for &axis in axes.iter().rev() {
                    xs = xs.unsqueeze(axis)?
                }
                values.insert(node.output[0].clone(), xs);
            }
            "Clip" => {
                let xs = get(&node.input[0])?;
                let xs = if let Some(mins) = get_opt(1) {
                    xs.broadcast_maximum(mins?)?
                } else {
                    xs.clone()
                };
                let xs = if let Some(maxs) = get_opt(2) {
                    xs.broadcast_minimum(maxs?)?
                } else {
                    xs.clone()
                };
                values.insert(node.output[0].clone(), xs);
            }
            "Gather" => {
                // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Gather
                let xs = get(&node.input[0])?;
                let indices = get(&node.input[1])?;
                let axis = get_attr_opt::<i64>(node, "axis")?.copied().unwrap_or(0);
                let axis = xs.normalize_axis(axis)?;

                // index_select does not support negative indices, so normalize them
                // to positive indices.
                let indices = &{
                    let zeros = Tensor::zeros(indices.shape(), indices.dtype(), indices.device())?;
                    let max = Tensor::new(xs.dims()[axis] as i64, indices.device())?
                        .to_dtype(indices.dtype())?;
                    let mask = indices.lt(&zeros)?;
                    mask.to_dtype(indices.dtype())?
                        .broadcast_mul(&max)?
                        .add(indices)?
                };

                // In Pytorch or Numpy this can be done by indexing the xs tensor using the indices
                // tensor directly, but candle does not support tensor indexing at the moment, so
                // some workarounds must be done.
                let xs = match indices.dims() {
                    [] => {
                        let index = indices.to_vec0::<i64>()? as usize;
                        xs.narrow(axis, index, 1)?.squeeze(axis)?
                    }
                    [_] => xs.index_select(indices, axis)?,
                    [first, _] => {
                        let mut v = Vec::with_capacity(*first);
                        for i in 0..*first {
                            v.push(xs.index_select(&indices.get(i)?, axis)?)
                        }
                        Tensor::stack(&v, axis)?
                    }
                    _ => {
                        // TODO: Provide an op to handle the ONNX generalized gather op ideally in a
                        // differentiable way.
                        todo!("implement gather for {xs:?} {indices:?} axis {axis}")
                    }
                };
                values.insert(node.output[0].clone(), xs);
            }
            // https://onnx.ai/onnx/operators/onnx__GatherElements.html#gatherelements
            // A Note to fellow lurkers:
            // The numpy based `gather_elements` implementation in `onnx` tests [here](https://github.com/onnx/onnx/blob/main/onnx/backend/test/case/node/gatherelements.py)
            // and examples is incorrect.
            // Use `torch.gather` for the validating/ verifying against the proper behaviour
            "GatherElements" => {
                let data = get(&node.input[0])?;
                let indices = get(&node.input[1])?;

                let rank = data.rank();
                if rank != indices.rank() {
                    bail!("indices must have same rank as input data. Data rank [{}] != indices rank [{}]", data.rank(), indices.rank());
                }

                let axis = {
                    let axis_i64 = get_attr_opt::<i64>(node, "axis")?.copied().unwrap_or(0);
                    let axis = data.normalize_axis(axis_i64)?;

                    if axis >= rank {
                        bail!(
                            "axis ({}) out of accepted range [-rank, rank-1] which was [-{rank}, {}]",
                            axis_i64,
                            rank - 1
                        )
                    }

                    axis
                };

                // index_select does not support negative indices, so normalize them
                // to positive indices.
                let indices = &{
                    let zeros = Tensor::zeros(indices.shape(), indices.dtype(), indices.device())?;
                    let max = Tensor::new(data.dims()[axis] as i64, indices.device())?
                        .to_dtype(indices.dtype())?;
                    let mask = indices.lt(&zeros)?;
                    mask.to_dtype(indices.dtype())?
                        .broadcast_mul(&max)?
                        .add(indices)?
                };

                values.insert(node.output[0].clone(), data.gather(indices, axis)?);
            }
            "Shape" => {
                // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Shape
                let xs = get(&node.input[0])?;
                let start = get_attr_opt::<i64>(node, "start")?.copied().unwrap_or(0);
                let end = get_attr_opt::<i64>(node, "end")?.copied().unwrap_or(-1);
                let start = xs.normalize_axis(start)?;
                let end = xs.normalize_axis(end)?;
                let mut dims = vec![];
                for idx in start..=end {
                    dims.push(xs.dim(idx)? as i64)
                }
                let dims = Tensor::from_vec(dims, xs.rank(), xs.device())?;
                values.insert(node.output[0].clone(), dims);
            }
            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Size
            "Size" => {
                let data = get(&node.input[0])?;
                let size: usize = data.dims().iter().product();
                let output = Tensor::from_slice(&[size as i64], (), data.device())?;
                values.insert(node.output[0].clone(), output);
            }
            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Sqrt
            "Sqrt" => {
                let xs = get(&node.input[0])?;
                let output = xs.sqrt()?;
                values.insert(node.output[0].clone(), output);
            }
            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Range
            "Range" => {
                let start = get(&node.input[0])?;
                let limit = get(&node.input[1])?;
                let delta = get(&node.input[2])?;

                macro_rules! arange_step {
                    ($t: ty) => {
                        Tensor::arange_step(
                            to_vec0_flexible::<$t>(start)?,
                            to_vec0_flexible::<$t>(limit)?,
                            to_vec0_flexible::<$t>(delta)?,
                            &Device::Cpu,
                        )?
                    };
                }

                let output = match start.dtype() {
                    DType::U8 => arange_step!(u8),
                    DType::U32 => arange_step!(u32),
                    DType::I64 => arange_step!(i64),
                    DType::BF16 => arange_step!(f32),
                    DType::F16 => arange_step!(f32),
                    DType::F32 => arange_step!(f32),
                    DType::F64 => arange_step!(f64),
                    DType::F8E4M3 => arange_step!(f32),
                    DType::I32
                    | DType::I16
                    | DType::F6E2M3
                    | DType::F6E3M2
                    | DType::F4
                    | DType::F8E8M0 => {
                        bail!("unsupported Range type i32/i16/f6e2m3/f6e3m2/f4/f8e8m0")
                    }
                };

                values.insert(node.output[0].clone(), output);
            }
            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Greater
            "Greater" => {
                let a = get(&node.input[0])?;
                let b = get(&node.input[1])?;

                let output = a.broadcast_gt(b)?;
                values.insert(node.output[0].clone(), output);
            }
            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Less
            "Less" => {
                let a = get(&node.input[0])?;
                let b = get(&node.input[1])?;

                let output = a.broadcast_lt(b)?;
                values.insert(node.output[0].clone(), output);
            }
            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#LessOrEqual
            "LessOrEqual" => {
                let a = get(&node.input[0])?;
                let b = get(&node.input[1])?;

                let output = a.broadcast_le(b)?;
                values.insert(node.output[0].clone(), output);
            }
            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#GreaterOrEqual
            "GreaterOrEqual" => {
                let a = get(&node.input[0])?;
                let b = get(&node.input[1])?;

                let output = a.broadcast_ge(b)?;
                values.insert(node.output[0].clone(), output);
            }
            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Log
            "Log" => {
                let a = get(&node.input[0])?;

                let output = a.log()?;
                values.insert(node.output[0].clone(), output);
            }
            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Min
            "Min" => {
                let mut output = get(&node.input[0])?.clone();
                for input in node.input.iter() {
                    let input = get(input)?;
                    output = output.broadcast_minimum(input)?
                }

                values.insert(node.output[0].clone(), output);
            }
            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Where
            "Where" => {
                let cond = get(&node.input[0])?;
                let a = get(&node.input[1])?;
                let b = get(&node.input[2])?;

                // where_cond requires that all inputs are the same shape.
                // In contrast, the Where op in ONNX only requires that they are broadcastable.
                let shape = broadcast_shape_from_many(&[cond.dims(), a.dims(), b.dims()])?;
                let cond = cond.broadcast_as(shape.clone())?;
                let a = a.broadcast_as(shape.clone())?;
                let b = b.broadcast_as(shape)?;
                let output = cond.where_cond(&a, &b)?;
                values.insert(node.output[0].clone(), output);
            }
            "Conv" => {
                // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Conv
                let dilations = get_attr_opt::<[i64]>(node, "dilations")?;
                let groups = get_attr_opt::<i64>(node, "group")?.copied().unwrap_or(1);
                let _kernel_shape = get_attr_opt::<[i64]>(node, "kernel_shape")?;
                let pads = get_attr_opt::<[i64]>(node, "pads")?;
                let strides = get_attr_opt::<[i64]>(node, "strides")?;
                let auto_pad = get_attr_opt::<str>(node, "auto_pad")?;
                match auto_pad {
                    None | Some("NOTSET") => (),
                    Some(s) => bail!("unsupported auto_pad {s}"),
                };
                let xs = get(&node.input[0])?;
                let ws = get(&node.input[1])?;
                let ys = match ws.rank() {
                    3 => {
                        let (pads, xs) = match pads {
                            None => (0, xs.clone()),
                            Some([p]) => (*p as usize, xs.clone()),
                            Some([p1, p2]) => {
                                if p1 != p2 {
                                    (0usize, xs.pad_with_zeros(2, *p1 as usize, *p2 as usize)?)
                                } else {
                                    (*p1 as usize, xs.clone())
                                }
                            }
                            Some(pads) => {
                                bail!("more pads than expected in conv1d {pads:?} {}", node.name)
                            }
                        };
                        let strides = match strides {
                            None => 1,
                            Some([p]) => *p as usize,
                            Some(s) => {
                                bail!("more strides than expected in conv1d {s:?} {}", node.name)
                            }
                        };
                        let dilations = match dilations {
                            None => 1,
                            Some([p]) => *p as usize,
                            Some(s) => {
                                bail!("more dilations than expected in conv1d {s:?} {}", node.name)
                            }
                        };
                        xs.conv1d(ws, pads, strides, dilations, groups as usize)?
                    }
                    4 => {
                        let (pads, xs) = match pads {
                            None => (0, xs.clone()),
                            Some([p]) => (*p as usize, xs.clone()),
                            Some(&[p1, p2, p3, p4]) => {
                                let p1 = p1 as usize;
                                let p2 = p2 as usize;
                                let p3 = p3 as usize;
                                let p4 = p4 as usize;
                                if p1 != p2 || p1 != p3 || p1 != p4 {
                                    (0, xs.pad_with_zeros(2, p1, p3)?.pad_with_zeros(3, p2, p4)?)
                                } else {
                                    (p1, xs.clone())
                                }
                            }
                            Some(pads) => {
                                bail!("more pads than expected in conv2d {pads:?} {}", node.name)
                            }
                        };
                        let (stride_h, stride_w) = match strides {
                            None => (1, 1),
                            Some([p]) => (*p as usize, *p as usize),
                            Some([p1, p2]) => (*p1 as usize, *p2 as usize),
                            Some(s) => {
                                bail!("more strides than expected in conv2d {s:?} {}", node.name)
                            }
                        };
                        let (dilation_h, dilation_w) = match dilations {
                            None => (1, 1),
                            Some([p]) => (*p as usize, *p as usize),
                            Some([p1, p2]) => (*p1 as usize, *p2 as usize),
                            Some(s) => {
                                bail!("more dilations than expected in conv2d {s:?} {}", node.name)
                            }
                        };
                        // If strides or dilations are different, we need to handle it specially
                        if stride_h != stride_w || dilation_h != dilation_w {
                            // For asymmetric strides, we do conv with stride 1, then subsample
                            let min_stride = stride_h.min(stride_w);
                            let min_dilation = dilation_h.min(dilation_w);
                            
                            // Perform conv2d with minimum stride/dilation
                            let conv_result = xs.conv2d(ws, pads, min_stride, min_dilation, groups as usize)?;
                            
                            // Now subsample to achieve the desired asymmetric stride
                            // conv_result shape: [N, C, H_out, W_out]
                            let (_, _, h_out, w_out) = conv_result.dims4()?;
                            
                            // Calculate subsampling factors
                            let subsample_h = stride_h / min_stride;
                            let subsample_w = stride_w / min_stride;
                            
                            // Create indices for subsampling
                            let h_indices: Vec<i64> = (0..h_out).step_by(subsample_h).map(|x| x as i64).collect();
                            let w_indices: Vec<i64> = (0..w_out).step_by(subsample_w).map(|x| x as i64).collect();
                            
                            let h_idx = Tensor::from_vec(h_indices.clone(), (h_indices.len(),), conv_result.device())?;
                            let w_idx = Tensor::from_vec(w_indices.clone(), (w_indices.len(),), conv_result.device())?;
                            
                            // Subsample along H and W dimensions
                            let result = conv_result.index_select(&h_idx, 2)?.index_select(&w_idx, 3)?;
                            result
                        } else {
                            xs.conv2d(ws, pads, stride_h, dilation_h, groups as usize)?
                        }
                    }
                    rank => bail!(
                        "unsupported rank for weight matrix {rank} in conv {}",
                        node.name
                    ),
                };
                let ys = if node.input.len() > 2 {
                    let bs = get(&node.input[2])?;
                    let mut bs_shape = vec![1; ys.rank()];
                    bs_shape[1] = bs.elem_count();
                    ys.broadcast_add(&bs.reshape(bs_shape)?)?
                } else {
                    ys
                };
                values.insert(node.output[0].clone(), ys);
            }
            "Concat" => {
                // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Concat
                let inputs = node
                    .input
                    .iter()
                    .map(|n| Ok(get(n.as_str())?.clone()))
                    .collect::<Result<Vec<Value>>>()?;
                let axis: i64 = *get_attr(node, "axis")?;
                if inputs.is_empty() {
                    bail!("empty concat")
                };
                // Find minimum rank among inputs and squeeze trailing singleton dims to match
                let min_rank = inputs.iter().map(|t| t.rank()).min().unwrap();
                let inputs: Vec<_> = inputs
                    .into_iter()
                    .map(|t| {
                        let mut t = t;
                        while t.rank() > min_rank {
                            let last_dim = t.rank() - 1;
                            if t.dims()[last_dim] == 1 {
                                t = t.squeeze(last_dim).unwrap_or(t);
                            } else {
                                break;
                            }
                        }
                        t
                    })
                    .collect();
                let axis = inputs[0].normalize_axis(axis)?;
                let output = Tensor::cat(&inputs, axis).map_err(|e| {
                    let shapes: Vec<_> = inputs.iter().map(|t| format!("{:?}", t.dims())).collect();
                    candle::Error::Msg(format!(
                        "Concat failed for node '{}': {} (input shapes: {:?})",
                        node.name, e, shapes
                    ))
                })?;
                values.insert(node.output[0].clone(), output);
            }
            "Abs" => {
                let input = get(&node.input[0])?;
                let output = input.abs()?;
                values.insert(node.output[0].clone(), output);
            }
            "Cos" => {
                let input = get(&node.input[0])?;
                let output = input.cos()?;
                values.insert(node.output[0].clone(), output);
            }
            "Sin" => {
                let input = get(&node.input[0])?;
                let output = input.sin()?;
                values.insert(node.output[0].clone(), output);
            }
            "Neg" => {
                let input = get(&node.input[0])?;
                // neg() not implemented for i64, work around with multiply by -1
                let output = if input.dtype() == DType::I64 {
                    let minus_one =
                        Tensor::new(&[-1i64], input.device())?.broadcast_as(input.shape())?;
                    input.mul(&minus_one)?
                } else {
                    input.neg()?
                };
                values.insert(node.output[0].clone(), output);
            }
            "Erf" => {
                let input = get(&node.input[0])?;
                let output = input.erf()?;
                values.insert(node.output[0].clone(), output);
            }
            "Tanh" => {
                let input = get(&node.input[0])?;
                let output = input.tanh()?;
                values.insert(node.output[0].clone(), output);
            }
            "Sigmoid" => {
                let input = get(&node.input[0])?;
                let output = candle_nn::ops::sigmoid(input)?;
                values.insert(node.output[0].clone(), output);
            }
            "Gelu" => {
                let input = get(&node.input[0])?;
                let output = input.gelu_erf()?;
                values.insert(node.output[0].clone(), output);
            }
            "Relu" => {
                let input = get(&node.input[0])?;
                let output = input.relu()?;
                values.insert(node.output[0].clone(), output);
            }
            "PRelu" => {
                // https://onnx.ai/onnx/operators/onnx__PRelu.html
                let input = get(&node.input[0])?;
                let slope = get(&node.input[1])?;

                let output = PReLU::new(slope.clone(), false).forward(input)?;
                values.insert(node.output[0].clone(), output);
            }
            "Ceil" => {
                let input = get(&node.input[0])?;
                let output = input.ceil()?;
                values.insert(node.output[0].clone(), output);
            }
            "Floor" => {
                let input = get(&node.input[0])?;
                let output = input.floor()?;
                values.insert(node.output[0].clone(), output);
            }
            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Round
            "Round" => {
                let input = get(&node.input[0])?;
                // Round to nearest even (banker's rounding)
                let output = input.round()?;
                values.insert(node.output[0].clone(), output);
            }
            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Constant
            "Constant" => {
                let value = match node.attribute.iter().find(|attr| attr.name == "value") {
                    None => {
                        // TODO: support sparse_value etc.
                        bail!("cannot find 'value' attr in 'Constant' for {}", node.name)
                    }
                    Some(value) => value,
                };
                let output = match value.r#type() {
                    AttributeType::Tensor => {
                        let t = value.t.as_ref().unwrap();
                        get_tensor(t, &node.name)?
                    }
                    rtype => bail!("unsupported 'value' type {rtype:?} for {}", node.name),
                };

                values.insert(node.output[0].clone(), output);
            }
            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Cast
            "Cast" => {
                let input = get(&node.input[0])?;
                let dt: i64 = *get_attr(node, "to")?;
                let dtype = match DataType::try_from(dt as i32) {
                    Ok(DataType::Int32) => DType::I64,
                    Ok(dt) => match dtype(dt) {
                        Some(dt) => dt,
                        None => {
                            bail!("unsupported 'to' value {dt:?} for cast {}", node.name)
                        }
                    },
                    Err(_) => {
                        bail!("unsupported 'to' value {dt:?} for cast {}", node.name)
                    }
                };
                let output = input.to_dtype(dtype)?;
                values.insert(node.output[0].clone(), output);
            }
            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#CumSum
            "CumSum" => {
                let exclusive = get_attr_opt::<i64>(node, "exclusive")?
                    .copied()
                    .unwrap_or(0);
                let reverse = get_attr_opt::<i64>(node, "reverse")?.copied().unwrap_or(0);
                if exclusive != 0 {
                    bail!("only exclusive == 0 is supported in CumSum")
                }
                if reverse != 0 {
                    bail!("only reverse == 0 is supported in CumSum")
                }
                let input = get(&node.input[0])?;
                let axis = to_vec0_flexible::<u32>(&get(&node.input[1])?.to_dtype(DType::U32)?)?;
                let output = input.cumsum(axis as usize)?;
                values.insert(node.output[0].clone(), output);
            }
            //  https://github.com/onnx/onnx/blob/main/docs/Operators.md#flatten
            "Flatten" => {
                let axis = get_attr_opt::<i64>(node, "axis")?.copied().unwrap_or(1) as usize;
                let input = get(&node.input[0])?;
                let first_part: usize = input.shape().dims().iter().take(axis).product();
                let end_index = input.shape().dims().iter().product::<usize>();
                let new_shape = (first_part, end_index / first_part);
                let output = input.reshape(new_shape)?;
                values.insert(node.output[0].clone(), output);
            }
            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#identity
            "Identity" => {
                let input = get(&node.input[0])?;
                values.insert(node.output[0].clone(), input.clone());
            }
            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#if
            "If" => {
                // protobuf encodes boolean false as 0 and true as 1
                let cond = to_scalar_flexible::<u8>(&get(&node.input[0])?.get(0)?)?;
                let attr_name = if cond != 0 {
                    "then_branch"
                } else {
                    "else_branch"
                };
                let sub_graph = get_attr::<GraphProto>(node, attr_name)?;
                if sub_graph.output.len() != node.output.len() {
                    bail!(
                        "If node {:?} is malformed: branch outputs ({}) don't match node outputs ({})",
                        node.name,
                        sub_graph.output.len(),
                        node.output.len()
                    );
                }
                let branch_out = simple_eval_(sub_graph, values)?;
                for (i, out) in node.output.iter().enumerate() {
                    values.insert(
                        out.clone(),
                        branch_out.get(&sub_graph.output[i].name).unwrap().clone(),
                    );
                }
            }
            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#pad
            "Pad" => {
                let mode = get_attr_opt(node, "mode")?.unwrap_or("constant");
                let data = get(&node.input[0])?;
                let pads = get(&node.input[1])?;
                // Third input is optional constant_value (default 0)
                let constant_value: f64 = if node.input.len() > 2 && !node.input[2].is_empty() {
                    let cv = get(&node.input[2])?;
                    to_scalar_flexible::<f64>(&cv.to_dtype(DType::F64)?)?
                } else {
                    0.0
                };
                // Fourth input is optional axes (not commonly used, ignore for now)
                if node.input.len() > 4 {
                    bail!(
                        "unsupported number of inputs {} for Pad node {:?}, expected at most 4",
                        node.input.len(),
                        node.name
                    );
                }
                if pads.rank() != 1 {
                    bail!("Pad expects 'pads' input to be 1D vector: {pads:?}");
                }
                if pads.dim(0).unwrap() != 2 * data.rank() {
                    bail!("Pad expects 'pads' input len to be 2 * rank of 'data' input: pads: {}, data rank: {}", pads, data.rank());
                }

                let pads = pads.to_vec1::<i64>()?;
                let (pads_pre, pads_post) = pads.split_at(pads.len() / 2);

                match mode {
                    "constant" => {
                        let mut out = data.clone();
                        for (i, _) in data.dims().iter().enumerate().rev() {
                            let pre = pads_pre[i] as usize;
                            let post = pads_post[i] as usize;
                            if pre == 0 && post == 0 {
                                continue;
                            }
                            // Create padding tensors with constant value
                            let mut pre_shape = out.dims().to_vec();
                            pre_shape[i] = pre;
                            let mut post_shape = out.dims().to_vec();
                            post_shape[i] = post;

                            let pre_pad = if pre > 0 {
                                Some(Tensor::full(constant_value, pre_shape, out.device())?.to_dtype(out.dtype())?)
                            } else {
                                None
                            };
                            let post_pad = if post > 0 {
                                Some(Tensor::full(constant_value, post_shape, out.device())?.to_dtype(out.dtype())?)
                            } else {
                                None
                            };

                            // Concatenate along axis i
                            let tensors: Vec<&Tensor> = [
                                pre_pad.as_ref(),
                                Some(&out),
                                post_pad.as_ref(),
                            ]
                            .into_iter()
                            .flatten()
                            .collect();

                            out = Tensor::cat(&tensors, i)?;
                        }
                        values.insert(node.output[0].clone(), out);
                    }
                    "reflect" => {
                        let mut out = data.clone();
                        for (i, &dim) in data.dims().iter().enumerate().rev() {
                            if pads_pre[i] == 0 && pads_post[i] == 0 {
                                continue;
                            }
                            fn zigzag(min: i64, max: i64) -> impl Iterator<Item = i64> {
                                std::iter::repeat((min..max).chain((min + 1..=max).rev())).flatten()
                            }
                            let idx = if dim > 1 {
                                let cycle_len = dim * 2 - 2;
                                let skip = cycle_len - ((pads_pre[i] as usize) % cycle_len);
                                let idx = zigzag(0, (dim - 1) as i64)
                                    .skip(skip)
                                    .take((pads_pre[i] as usize) + dim + (pads_post[i] as usize));
                                Tensor::from_iter(idx, out.device())?
                            } else {
                                Tensor::full(0i64, (dim,), out.device())?
                            };

                            out = out.index_select(&idx, i)?;
                        }

                        values.insert(node.output[0].clone(), out);
                    }
                    _ => bail!(
                        "unsupported 'mode' value {mode:?} for Pad node {:?}",
                        node.name
                    ),
                }
            }
            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#slice
            "Slice" => {
                let data = get(&node.input[0])?;
                let starts = get(&node.input[1])?;
                let ends = get(&node.input[2])?;
                let default_axes;
                let default_steps;
                let axes: &Tensor;
                let steps: &Tensor;
                // If axes are omitted, they are set to [0, ..., r-1]. If steps are omitted,
                // they are set to [1, ..., 1] of length len(starts)
                match node.input.len() {
                    3 => {
                        let len = starts.dims()[0];
                        default_axes = Some(Tensor::arange(0, len as i64, starts.device())?);
                        axes = default_axes.as_ref().unwrap();
                        default_steps = Some(Tensor::ones((len,), DType::I64, starts.device())?);
                        steps = default_steps.as_ref().unwrap();
                    }
                    4 => {
                        let len = starts.dims()[0];
                        axes = get(&node.input[3])?;
                        default_steps = Some(Tensor::ones((len,), DType::I64, starts.device())?);
                        steps = default_steps.as_ref().unwrap();
                    }
                    5 => {
                        steps = get(&node.input[4])?;
                        axes = get(&node.input[3])?;
                    }
                    _ => bail!(
                        "Slice node is invalid, expected 3-5 inputs, got {}: {:?}",
                        node.input.len(),
                        node
                    ),
                }

                let mut out = data.clone();
                for (i, axis) in axes.to_vec1::<i64>()?.into_iter().enumerate() {
                    // All negative elements of axes are made non-negative by
                    // adding r to them, where r = rank(input).
                    let axis = if axis < 0 {
                        axis + data.rank() as i64
                    } else {
                        axis
                    } as usize;

                    let data_dim = data.dims()[axis] as i64;
                    let mut s = to_scalar_flexible::<i64>(&starts.get(i)?)?;
                    let mut e = to_scalar_flexible::<i64>(&ends.get(i)?)?;
                    // All negative values in starts[i] and ends[i] have
                    // dims[axes[i]] added to them, where dims are the
                    // dimensions of input.
                    if s < 0 {
                        s += data_dim;
                    }
                    if e < 0 {
                        e += data_dim;
                    }

                    let p = to_scalar_flexible::<i64>(&steps.get(i)?)?;
                    // starts[i] is clamped into the range [0, dims[axes[i]]]
                    // for positive stepping and [0, dims[axes[i]]-1] for
                    // negative stepping.
                    // for positive stepping ends[axes[i]] is clamped to
                    // [0, dims[axes[i]]], while for negative stepping it is
                    // clamped to [-1, dims[axes[i]]-1].
                    if p >= 0 {
                        s = s.clamp(0, data_dim);
                        e = e.clamp(0, data_dim);
                    } else {
                        s = s.clamp(0, data_dim - 1);
                        e = e.clamp(-1, data_dim - 1);
                    }

                    let indexes = Tensor::arange_step(s, e, p, data.device())?;
                    out = out.contiguous()?.index_select(&indexes, axis)?
                }
                values.insert(node.output[0].clone(), out);
            }
            // https://onnx.ai/onnx/operators/onnx__ReduceMax.html#reducemax
            "ReduceMax" => {
                let input = get(&node.input[0])?;
                let axes = get_opt(1);
                let keepdims = get_attr_opt::<i64>(node, "keepdims")?.copied().unwrap_or(1) == 1;

                let axes = if let Some(Ok(axes)) = axes {
                    // Satisfies version 18+
                    axes.to_vec1::<i64>().ok()
                } else if let Ok(Some(axes)) = get_attr_opt::<[i64]>(node, "axes") {
                    // Backward compatibility with version 13 and below
                    Some(axes.to_vec())
                } else {
                    None
                };

                let axes = if let Some(axes) = axes {
                    let rank = input.rank();
                    let mut axes_set = HashSet::new();

                    let mut axes = axes
                        .iter()
                        .map(|a| {
                            let axis = if *a < 0 {
                                (rank as i64 + *a) as usize
                            } else {
                                *a as usize
                            };

                            axes_set.insert(axis);
                            axis
                        })
                        .collect::<Vec<_>>();

                    if axes_set.len() < axes.len() {
                        bail!("Duplicate value in 'axes'");
                    }

                    if axes.len() > 1 {
                        axes.sort();
                    }

                    Some(axes)
                } else {
                    None
                };

                // TODO: Handle empty set
                // Definition:
                // "Reduction over an empty set of values yields minus infinity (if supported by the datatype) or the minimum value of the data type otherwise"
                // For now, this will throw an error
                if input.elem_count() == 0 {
                    bail!("reduction over zero-size tensor not supported");
                }

                let output = if let Some(axes) = axes {
                    let mut result = input.clone();
                    for &axis in axes.iter().rev() {
                        result = if keepdims {
                            result.max_keepdim(axis)?
                        } else {
                            result.max(axis)?
                        }
                    }

                    result
                } else {
                    // If `axes` is empty and `noop_with_empty_axes` is set to `true (1)`
                    // ""input tensor will not be reduced,and the output tensor would be equivalent to input tensor.""
                    if get_attr_opt::<i64>(node, "noop_with_empty_axes")?.copied() == Some(1) {
                        input.clone()
                    } else {
                        let mut result = input.flatten_all()?;
                        if keepdims {
                            result = result.max_keepdim(0)?;
                            // If keepdims is true, reshape to match input dimensions
                            let shape = vec![1; input.rank()];
                            result.reshape(shape)?
                        } else {
                            result.max(0)?
                        }
                    }
                };

                values.insert(node.output[0].clone(), output);
            }
            // https://onnx.ai/onnx/operators/onnx__ReduceMean.html#reducemean-13
            // TODO: This version is only compatible with ReduceMean V13 and below.
            "ReduceMean" => {
                let input = get(&node.input[0])?;
                let axes = get_attr_opt::<[i64]>(node, "axes")?;
                let keepdims = get_attr_opt::<i64>(node, "keepdims")?.copied().unwrap_or(1);

                let n_dims = input.dims().len();

                let axes: Vec<usize> = if let Some(axes) = axes {
                    axes.iter()
                        .map(|e| (if e < &0 { (n_dims as i64) + *e } else { *e }) as usize)
                        .collect()
                } else {
                    (0..n_dims).collect()
                };
                let output = if keepdims == 1 {
                    input.mean_keepdim(axes)?
                } else {
                    input.mean(axes)?
                };
                values.insert(node.output[0].clone(), output);
            }
            // https://onnx.ai/onnx/operators/onnx__ReduceMin.html#reducemin
            "ReduceMin" => {
                let input = get(&node.input[0])?;
                let axes = get_opt(1);
                let keepdims = get_attr_opt::<i64>(node, "keepdims")?.copied().unwrap_or(1) == 1;

                let axes = if let Some(Ok(axes)) = axes {
                    // Satisfies version 18+
                    axes.to_vec1::<i64>().ok()
                } else if let Ok(Some(axes)) = get_attr_opt::<[i64]>(node, "axes") {
                    // Backward compatibility with version 13 and below
                    Some(axes.to_vec())
                } else {
                    None
                };

                let axes = if let Some(axes) = axes {
                    let rank = input.rank();
                    let mut axes_set = HashSet::new();

                    let mut axes = axes
                        .iter()
                        .map(|a| {
                            let axis = if *a < 0 {
                                (rank as i64 + *a) as usize
                            } else {
                                *a as usize
                            };

                            axes_set.insert(axis);
                            axis
                        })
                        .collect::<Vec<_>>();

                    if axes_set.len() < axes.len() {
                        bail!("Duplicate value in 'axes'");
                    }

                    if axes.len() > 1 {
                        axes.sort();
                    }

                    Some(axes)
                } else {
                    None
                };

                // TODO: Handle empty set
                // Definition:
                // "Reduction over an empty set of values yields positive infinity (if supported by the datatype) or the max value of the data type otherwise"
                // For now, this will throw an error
                if input.elem_count() == 0 {
                    bail!("reduction over zero-size tensor not supported");
                }

                let output = if let Some(axes) = axes {
                    let mut result = input.clone();
                    for &axis in axes.iter().rev() {
                        result = if keepdims {
                            result.min_keepdim(axis)?
                        } else {
                            result.min(axis)?
                        }
                    }

                    result
                } else {
                    // If `axes` is empty and `noop_with_empty_axes` is set to `true (1)`
                    // ""input tensor will not be reduced,and the output tensor would be equivalent to input tensor.""
                    if get_attr_opt::<i64>(node, "noop_with_empty_axes")?.copied() == Some(1) {
                        input.clone()
                    } else {
                        let mut result = input.flatten_all()?;
                        if keepdims {
                            result = result.min_keepdim(0)?;
                            // If keepdims is true, reshape to match input dimensions
                            let shape = vec![1; input.rank()];
                            result.reshape(shape)?
                        } else {
                            result.min(0)?
                        }
                    }
                };

                values.insert(node.output[0].clone(), output);
            }
            //https://github.com/onnx/onnx/blob/main/docs/Operators.md#Split
            // Version 18 impl
            "Split" => {
                let input_tensor = get(&node.input[0])?;
                let axis = get_attr_opt::<i64>(node, "axis")?.copied().unwrap_or(0);
                let axis = input_tensor.normalize_axis(axis)?;

                // Determine split sizes
                let splits = if node.input.len() > 1 {
                    // If the split tensor is provided, use it to determine sizes
                    let split_tensor = get(&node.input[1])?.to_vec1::<i64>()?;
                    split_tensor.iter().map(|&x| x as usize).collect::<Vec<_>>()
                } else {
                    let num_outputs = if let Some(&num_outputs_attrib) =
                        get_attr_opt::<i64>(node, "num_outputs")?
                    {
                        num_outputs_attrib as usize
                    } else {
                        node.output.len()
                    };

                    let input_dim = input_tensor.dim(axis)?;

                    let mut split_sizes =
                        vec![input_dim / num_outputs as usize; num_outputs as usize];
                    let remainder = input_dim % num_outputs as usize;
                    if remainder > 0 {
                        // If there's a remainder, add it to the last split size
                        split_sizes[num_outputs as usize - 1] += remainder;
                    }

                    split_sizes
                };

                // Perform the split operation
                let mut outputs = vec![];
                let mut start = 0;
                for &size in &splits {
                    let end = start + size;
                    let slice = input_tensor.narrow(axis, start, size)?;
                    outputs.push(slice);
                    start = end;
                }

                // Insert the split outputs into the values map
                for (output, slice) in node.output.iter().zip(outputs.into_iter()) {
                    values.insert(output.clone(), slice);
                }
            }
            //https://github.com/onnx/onnx/blob/main/docs/Operators.md#Expand
            // Version 13 impl
            "Expand" => {
                // unlike broadcast_to, expand allows for the output shape to
                // be different from the specified shape.
                let input_tensor = get(&node.input[0])?;
                let input_shape = get(&node.input[1])?;

                // Check that the shape tensor is 1D
                if input_shape.rank() != 1 {
                    bail!(
                        "Expand expects 'shape' input to be 1D tensor: {:?}",
                        input_shape
                    );
                }
                let input_tensor_dims = input_tensor.dims();
                let input_shape_dims = input_shape
                    .to_vec1::<i64>()?
                    .into_iter()
                    .map(|x| x as usize)
                    .collect::<Vec<_>>();

                let target_shape = broadcast_shape(input_tensor_dims, input_shape_dims.as_slice())?;

                let expanded_tensor = input_tensor.broadcast_as(target_shape)?;

                values.insert(node.output[0].clone(), expanded_tensor);
            }
            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Tile
            "Tile" => {
                let input = get(&node.input[0])?;
                let repeats = get(&node.input[1])?.to_vec1::<i64>()?;

                let mut result = input.clone();
                for (dim, &repeat) in repeats.iter().enumerate() {
                    if repeat > 1 {
                        let repeat = repeat as usize;
                        let tensors: Vec<_> = (0..repeat).map(|_| result.clone()).collect();
                        result = Tensor::cat(&tensors, dim)?;
                    }
                }
                values.insert(node.output[0].clone(), result);
            }
            //https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceSum
            // Version 13 impl
            "ReduceSum" => {
                let input = get(&node.input[0])?;
                let axes = get_opt(1);
                let keepdims = get_attr_opt::<i64>(node, "keepdims")?.copied().unwrap_or(1);
                let noop_with_empty_axes = get_attr_opt::<i64>(node, "noop_with_empty_axes")?
                    .copied()
                    .unwrap_or(0);

                let axes = match axes {
                    Some(Ok(axes)) => axes
                        .to_vec1::<i64>()?
                        .into_iter()
                        .map(|x| input.normalize_axis(x).unwrap_or(x as usize))
                        .collect::<Vec<_>>(),
                    Some(Err(_)) | None => {
                        if noop_with_empty_axes == 1 {
                            vec![]
                        } else {
                            (0..input.rank()).collect()
                        }
                    }
                };

                let output = if keepdims == 1 {
                    input.sum_keepdim(axes)?
                } else {
                    input.sum(axes)?
                };

                values.insert(node.output[0].clone(), output);
            }
            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceProd
            "ReduceProd" => {
                let input = get(&node.input[0])?;
                let axes = get_opt(1);
                let keepdims = get_attr_opt::<i64>(node, "keepdims")?.copied().unwrap_or(1);
                let noop_with_empty_axes = get_attr_opt::<i64>(node, "noop_with_empty_axes")?
                    .copied()
                    .unwrap_or(0);

                let axes: Vec<usize> = match axes {
                    Some(Ok(axes)) => axes
                        .to_vec1::<i64>()?
                        .into_iter()
                        .map(|x| input.normalize_axis(x).unwrap_or(x as usize))
                        .collect(),
                    Some(Err(_)) | None => {
                        if noop_with_empty_axes == 1 {
                            vec![]
                        } else {
                            (0..input.rank()).collect()
                        }
                    }
                };

                // Candle doesn't have a direct prod method, so we use log + sum + exp
                // But this doesn't work for negative numbers or zeros
                // Instead, we iterate over axes and use a manual product
                let mut output = input.clone();
                let mut sorted_axes = axes.clone();
                sorted_axes.sort();
                sorted_axes.reverse(); // Process from highest to lowest to avoid index shifting
                
                for axis in sorted_axes {
                    // Get the size of this axis
                    let axis_size = output.dim(axis)?;
                    if axis_size == 0 {
                        continue;
                    }
                    // Compute product along this axis
                    let mut prod = output.narrow(axis, 0, 1)?;
                    for i in 1..axis_size {
                        let slice = output.narrow(axis, i, 1)?;
                        prod = prod.mul(&slice)?;
                    }
                    if keepdims == 1 {
                        output = prod;
                    } else {
                        output = prod.squeeze(axis)?;
                    }
                }

                values.insert(node.output[0].clone(), output);
            }
            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceL2
            // Version 18 impl
            "ReduceL2" => {
                let input = get(&node.input[0])?;
                let axes = get_opt(1);
                let keepdims = get_attr_opt::<i64>(node, "keepdims")?.copied().unwrap_or(1);
                let noop_with_empty_axes = get_attr_opt::<i64>(node, "noop_with_empty_axes")?
                    .copied()
                    .unwrap_or(0);

                let input_sq = input.sqr()?;

                let axes = match axes {
                    Some(axes) => axes?
                        .to_vec1::<i64>()?
                        .into_iter()
                        .map(|x| x as usize)
                        .collect::<Vec<_>>(),
                    None => {
                        if noop_with_empty_axes == 1 {
                            vec![]
                        } else {
                            (0..input_sq.rank()).collect()
                        }
                    }
                };

                let output = if keepdims == 1 {
                    input_sq.sum_keepdim(axes)?.sqrt()?
                } else {
                    input_sq.sum(axes)?.sqrt()?
                };

                values.insert(node.output[0].clone(), output);
            }
            random_type @ ("RandomUniform" | "RandomNormal") => {
                let dt: i64 = get_attr_opt(node, "dtype")?.copied().unwrap_or(1); // 1 is float
                                                                                  // type by
                                                                                  // default
                let dtype = match DataType::try_from(dt as i32) {
                    Ok(dt) => match dtype(dt) {
                        Some(DType::U8 | DType::U32 | DType::I64) => {
                            bail!(
                                "unsupported 'dtype' value {dt:?}, only floats are allowed, for {random_type} {}",
                                node.name
                            )
                        }
                        Some(dt) => dt,
                        None => {
                            bail!(
                                "unsupported 'dtype' value {dt:?} for {random_type} {}",
                                node.name
                            )
                        }
                    },
                    Err(_) => {
                        bail!(
                            "unsupported 'dtype' value {dt:?} for {random_type} {}",
                            node.name
                        )
                    }
                };
                let seed: Option<f32> = get_attr_opt(node, "seed")?.copied();
                if seed.is_some() {
                    bail!("seed for {random_type} is currently not supported")
                };
                let shape: Vec<usize> = get_attr::<[i64]>(node, "shape")?
                    .iter()
                    .map(|x| *x as usize)
                    .collect();
                let output = if random_type == "RandomUniform" {
                    let low: f32 = get_attr_opt(node, "low")?.copied().unwrap_or(0.0);
                    let high: f32 = get_attr_opt(node, "high")?.copied().unwrap_or(1.0);
                    Tensor::rand(low, high, shape, &Device::Cpu)?.to_dtype(dtype)?
                } else {
                    let mean: f32 = get_attr_opt(node, "mean")?.copied().unwrap_or(0.0);
                    let scale: f32 = get_attr_opt(node, "scale")?.copied().unwrap_or(1.0);
                    Tensor::randn(mean, scale, shape, &Device::Cpu)?.to_dtype(dtype)?
                };
                values.insert(node.output[0].clone(), output);
            }
            "ArgMin" => {
                let input = get(&node.input[0])?;
                let axis_i64: i64 = get_attr_opt(node, "axis")?.copied().unwrap_or(0);
                let rank_i64: i64 = input.rank().try_into().unwrap();
                if axis_i64 < -rank_i64 || axis_i64 >= rank_i64 {
                    bail!(
                        "axis ({}) out of accepted range [-rank, rank-1] which was [{}, {}]",
                        axis_i64,
                        -rank_i64,
                        rank_i64 - 1
                    )
                }
                let axis = input.normalize_axis(axis_i64)?;
                let keepdims: i64 = get_attr_opt(node, "keepdims")?.copied().unwrap_or(1);
                let select_last_index: i64 = get_attr_opt(node, "select_last_index")?
                    .copied()
                    .unwrap_or(0);
                if select_last_index == 1 {
                    bail!("select_last_index for ArgMin is currently not supported")
                }
                let output = if keepdims == 1 {
                    input.argmin_keepdim(axis)?
                } else {
                    input.argmin(axis)?
                }
                .to_dtype(DType::I64)?;
                values.insert(node.output[0].clone(), output);
            }
            "ArgMax" => {
                let input = get(&node.input[0])?;
                let axis_i64: i64 = get_attr_opt(node, "axis")?.copied().unwrap_or(0);
                let rank_i64: i64 = input.rank().try_into().unwrap();
                if axis_i64 < -rank_i64 || axis_i64 >= rank_i64 {
                    bail!(
                        "axis ({}) out of accepted range [-rank, rank-1] which was [{}, {}]",
                        axis_i64,
                        -rank_i64,
                        rank_i64 - 1
                    )
                }
                let axis = input.normalize_axis(axis_i64)?;
                let keepdims: i64 = get_attr_opt(node, "keepdims")?.copied().unwrap_or(1);
                let select_last_index: i64 = get_attr_opt(node, "select_last_index")?
                    .copied()
                    .unwrap_or(0);
                if select_last_index == 1 {
                    bail!("select_last_index for ArgMin is currently not supported")
                }
                let output = if keepdims == 1 {
                    input.argmax_keepdim(axis)?
                } else {
                    input.argmax(axis)?
                }
                .to_dtype(DType::I64)?;
                values.insert(node.output[0].clone(), output);
            }
            "LeakyRelu" => {
                let input = get(&node.input[0])?;
                let dt = input.dtype();
                match dt {
                    DType::U8
                    | DType::U32
                    | DType::I64
                    | DType::I32
                    | DType::I16
                    | DType::F6E2M3
                    | DType::F6E3M2
                    | DType::F4
                    | DType::F8E8M0 => {
                        bail!(
                            "unsupported dtype {}, only float types are allowed for LeakyRelu",
                            dt.as_str()
                        )
                    }
                    DType::BF16 | DType::F16 | DType::F32 | DType::F64 | DType::F8E4M3 => {}
                }
                let alpha = get_attr_opt::<f32>(node, "alpha")?.copied().unwrap_or(0.01);
                let output = candle_nn::ops::leaky_relu(input, alpha.into())?;
                values.insert(node.output[0].clone(), output);
            }
            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Gemm
            "Gemm" => {
                let a = get(&node.input[0])?;
                let b = get(&node.input[1])?;
                let c = get(&node.input[2])?;

                let alpha = get_attr_opt::<f32>(node, "alpha")?.copied().unwrap_or(1.0);
                let beta = get_attr_opt::<f32>(node, "beta")?.copied().unwrap_or(1.0);

                let alpha = Tensor::full(alpha, a.shape(), &Device::Cpu)?;
                let beta = Tensor::full(beta, c.shape(), &Device::Cpu)?;

                let trans_a = get_attr_opt::<i64>(node, "transA")?.copied().unwrap_or(0);
                let trans_b = get_attr_opt::<i64>(node, "transB")?.copied().unwrap_or(0);

                let a = if trans_a == 0 { a.clone() } else { a.t()? };
                let b = if trans_b == 0 { b.clone() } else { b.t()? };

                let output = a
                    .broadcast_mul(&alpha)?
                    .broadcast_matmul(&b)?
                    .broadcast_add(&c.broadcast_mul(&beta)?)?;
                values.insert(node.output[0].clone(), output);
            }
            "LSTM" => {
                let direction = get_attr_opt(node, "direction")?.unwrap_or("forward");
                if direction != "forward" {
                    bail!("LSTM currently only supports direction == \"forward\"");
                }
                let num_directions = if direction == "bidirectional" { 2 } else { 1 };
                let hidden_size: i64 = get_attr(node, "hidden_size").copied()?;
                let input_forget = get_attr_opt(node, "input_forget")?.copied().unwrap_or(0);
                if input_forget != 0 {
                    bail!("LSTM currently only supports input_forget == 0");
                }
                let activations_default = vec![
                    "Sigmoid".to_string(),
                    "Tanh".to_string(),
                    "Tanh".to_string(),
                ];
                let activations = get_attr_opt_owned::<Vec<String>>(node, "activations")?
                    .unwrap_or(activations_default.clone());
                if activations != activations_default {
                    bail!("LSTM currently only supports default activations ({activations_default:?})");
                }
                // activation_alpha and activation_beta don't apply to (Sigmoid, Tanh, Tanh) so ignoring them is okay
                if get_attr_opt::<f32>(node, "clip")?.is_some() {
                    bail!("LSTM does not currently support clip attribute");
                }

                // The shape format of inputs X, initial_h and outputs Y, Y_h.
                // If 0, the following shapes are expected:
                //     X.shape = [seq_length, batch_size, input_size],
                //     Y.shape = [seq_length, num_directions, batch_size, hidden_size],
                //     initial_h.shape = Y_h.shape = [num_directions, batch_size, hidden_size].
                // If 1, the following shapes are expected:
                //     X.shape = [batch_size, seq_length, input_size],
                //     Y.shape = [batch_size, seq_length, num_directions, hidden_size],
                //     initial_h.shape = Y_h.shape = [batch_size, num_directions, hidden_size].
                let layout = get_attr_opt(node, "layout")?.copied().unwrap_or(0);
                if layout != 0 {
                    bail!("LSTM currently only supports layout == 0");
                }

                // The input sequences packed (and potentially padded) into one 3-D tensor
                // with the shape of `[seq_length, batch_size, input_size]`.
                let x = get(&node.input[0])?;
                // XXX: depends on layout
                let (seq_length, batch_size, input_size) = x.dims3()?;
                // The weight tensor for the gates.
                // Concatenation of `W[iofc]` and `WB[iofc]` (if bidirectional) along dimension 0.
                // The tensor has shape `[num_directions, 4*hidden_size, input_size]`.
                let w = get(&node.input[1])?;
                // The recurrence weight tensor.
                // Concatenation of `R[iofc]` and `RB[iofc]` (if bidirectional) along dimension 0.
                // This tensor has shape `[num_directions, 4*hidden_size, hidden_size]`.
                let r = get(&node.input[2])?;

                // The bias tensor for input gate.
                // Concatenation of `[Wb[iofc], Rb[iofc]]`, and `[WBb[iofc], RBb[iofc]]` (if bidirectional) along dimension 0.
                // This tensor has shape `[num_directions, 8*hidden_size]`.
                // Optional: If not specified - assumed to be 0.
                let b_default: Tensor;
                let b = match get_opt(3) {
                    Some(n) => n?,
                    None => {
                        b_default = Tensor::zeros(
                            (num_directions, 8 * hidden_size as usize),
                            DType::F32,
                            x.device(),
                        )?;
                        &b_default
                    }
                };

                // Optional tensor specifying lengths of the sequences in a batch.
                // If not specified - assumed all sequences in the batch to have length `seq_length`.
                // It has shape `[batch_size]`.
                let seq_lens_default: Tensor;
                let seq_lens = match get_opt(4) {
                    Some(n) => n?,
                    None => {
                        seq_lens_default =
                            Tensor::full(seq_length as i64, (batch_size,), x.device())?;
                        &seq_lens_default
                    }
                };
                let seq_lens_is_default =
                    (seq_lens.to_vec1::<i64>()?.iter()).all(|e| *e as usize == seq_length);
                if !seq_lens_is_default {
                    bail!("LSTM currently only supports default value of seq_lens");
                }

                // Optional initial value of the hidden. If not specified - assumed to be 0.
                // It has shape `[num_directions, batch_size, hidden_size]`.
                let initial_h_default: Tensor;
                let initial_h = match get_opt(5) {
                    Some(n) => n?,
                    _ => {
                        initial_h_default = Tensor::zeros(
                            (num_directions, batch_size, hidden_size as usize),
                            DType::F32,
                            x.device(),
                        )?;
                        &initial_h_default
                    }
                };

                // Optional initial value of the cell.
                // If not specified - assumed to be 0.
                // It has shape `[num_directions, batch_size, hidden_size]`.
                let initial_c_default: Tensor;
                let initial_c = match node.input.get(6) {
                    Some(n) if !n.is_empty() => get(n)?,
                    _ => {
                        initial_c_default = Tensor::zeros(
                            (num_directions, batch_size, hidden_size as usize),
                            DType::F32,
                            x.device(),
                        )?;
                        &initial_c_default
                    }
                };

                // The weight tensor for peepholes.
                // Concatenation of `P[iof]` and `PB[iof]` (if bidirectional) along dimension 0.
                // It has shape `[num_directions, 3*hidde_size]`. Optional: If not specified - assumed to be 0.
                let p_default = Tensor::zeros(
                    (num_directions, 3 * hidden_size as usize),
                    DType::F32,
                    x.device(),
                )?;
                let p = get_opt(7).unwrap_or(Ok(&p_default))?;
                let p_is_zeros = (p.to_vec2::<f32>()?.iter()).all(|v| v.iter().all(|e| *e == 0.0));
                if !p_is_zeros {
                    bail!(
                        "LSTM currently only supports default value of p (a Tensor of all zeroes)"
                    );
                }

                // these all have [num_directions, ...] shapes
                let w = w.get(0)?; // w[iofc] has shape [4*hidden_size, input_size]
                let r = r.get(0)?; // r[iofc] has shape [4*hidden_size, hidden_size]
                let b = b.get(0)?; // concat of [wb[iofc],rb[iofc]] has shape [8*hidden_size]
                let idx_wb = Tensor::arange(0, 4 * hidden_size, x.device())?;
                let idx_rb = Tensor::arange(4 * hidden_size, 8 * hidden_size, x.device())?;
                let wb = b.index_select(&idx_wb, 0)?;
                let rb = b.index_select(&idx_rb, 0)?;
                let c = initial_c.get(0)?;
                let h = initial_h.get(0)?;

                // w, r, wb, rb are all iofc but lstm expects ifco
                // so we need to move some stuff around
                let idx_i = Tensor::arange(0, hidden_size, x.device())?;
                let idx_o = Tensor::arange(hidden_size, 2 * hidden_size, x.device())?;
                let idx_f = Tensor::arange(2 * hidden_size, 3 * hidden_size, x.device())?;
                let idx_c = Tensor::arange(3 * hidden_size, 4 * hidden_size, x.device())?;
                let idx_ifco = Tensor::cat(&[&idx_i, &idx_f, &idx_c, &idx_o], 0)?;
                let w = w.index_select(&idx_ifco, 0)?;
                let r = r.index_select(&idx_ifco, 0)?;
                let wb = wb.index_select(&idx_ifco, 0)?;
                let rb = rb.index_select(&idx_ifco, 0)?;
                let vmap = candle_nn::VarMap::new();
                vmap.data().lock().unwrap().extend([
                    ("weight_ih_l0".to_string(), candle::Var::from_tensor(&w)?),
                    ("weight_hh_l0".to_string(), candle::Var::from_tensor(&r)?),
                    ("bias_ih_l0".to_string(), candle::Var::from_tensor(&wb)?),
                    ("bias_hh_l0".to_string(), candle::Var::from_tensor(&rb)?),
                ]);
                use candle_nn::rnn::RNN as _;
                let lstm = candle_nn::rnn::lstm(
                    input_size,
                    hidden_size as usize,
                    candle_nn::rnn::LSTMConfig::default(),
                    candle_nn::VarBuilder::from_varmap(&vmap, w.dtype(), w.device()),
                )?;

                let mut lstm_state = candle_nn::rnn::LSTMState::new(h, c);
                let mut h_acc = if node.output.first().map(String::as_str).unwrap_or("") != "" {
                    Some(vec![])
                } else {
                    None
                };
                for t in 0..seq_length {
                    let x = x.get(t)?;
                    lstm_state = lstm.step(&x, &lstm_state)?;
                    if let Some(h_acc) = &mut h_acc {
                        h_acc.push(lstm_state.clone());
                    }
                }

                assert_eq!(num_directions, 1, "if support for bidirectional is ever added, outputs will have to be concatenated, not simply reshaped");
                if let Some(name) = node.output.first() {
                    let h_acc = h_acc.as_ref().unwrap();
                    let h_acc = lstm.states_to_tensor(h_acc)?;
                    let h_acc = h_acc.reshape((
                        seq_length,
                        num_directions,
                        batch_size,
                        hidden_size as usize,
                    ))?;
                    values.insert(name.clone(), h_acc);
                }
                if let Some(name) = node.output.get(1) {
                    values.insert(
                        name.clone(),
                        lstm_state.h().reshape((
                            num_directions,
                            batch_size,
                            hidden_size as usize,
                        ))?,
                    );
                }
                if let Some(name) = node.output.get(2) {
                    values.insert(
                        name.clone(),
                        lstm_state.c().reshape((
                            num_directions,
                            batch_size,
                            hidden_size as usize,
                        ))?,
                    );
                }
            }
            "RNN" => {
                // activation_alpha and activation_beta don't apply to (Tanh, Tanh) so ignoring them is okay
                let activations_default = vec!["Tanh".to_string(), "Tanh".to_string()];
                let activations = get_attr_opt_owned::<Vec<String>>(node, "activations")?
                    .unwrap_or(activations_default.clone());
                let clip = get_attr_opt::<f32>(node, "clip")?.copied();
                if clip.is_some() {
                    bail!("RNN does not currently support clip attribute");
                }
                let direction = get_attr_opt(node, "direction")?.unwrap_or("forward");
                if direction != "forward" {
                    bail!("RNN currently only supports direction == \"forward\"");
                }
                let num_directions = if direction == "bidirectional" { 2 } else { 1 };
                let hidden_size: i64 = get_attr(node, "hidden_size").copied()?;

                // The shape format of inputs X, initial_h and outputs Y, Y_h.
                // If 0, the following shapes are expected:
                //    X.shape = [seq_length, batch_size, input_size],
                //    Y.shape = [seq_length, num_directions, batch_size, hidden_size],
                //    initial_h.shape = Y_h.shape = [num_directions, batch_size, hidden_size].
                // If 1, the following shapes are expected:
                //    X.shape = [batch_size, seq_length, input_size],
                //    Y.shape = [batch_size, seq_length, num_directions, hidden_size],
                //    initial_h.shape = Y_h.shape = [batch_size, num_directions, hidden_size].
                let layout = get_attr_opt(node, "layout")?.copied().unwrap_or(0);
                if layout != 0 {
                    bail!("RNN currently only supports layout == 0");
                }

                // The input sequences packed (and potentially padded) into one 3-D tensor
                // with the shape of `[seq_length, batch_size, input_size]`.
                let x = get(&node.input[0])?;
                // XXX: depends on layout
                let (seq_length, batch_size, _) = x.dims3()?;
                // The weight tensor for the input gate.
                // Concatenation of `Wi` and `WBi` (if bidirectional).
                // The tensor has shape `[num_directions, hidden_size, input_size]`.
                let w = get(&node.input[1])?;
                // The recurrence weight tensor.
                // Concatenation of `Ri` and `RBi` (if bidirectional).
                // This tensor has shape `[num_directions, hidden_size, hidden_size]`.
                let r = get(&node.input[2])?;

                // The bias tensor for input gate.
                // Concatenation of `[Wbi, Rbi]` and `[WBbi, RBbi]` (if bidirectional).
                // This tensor has shape `[num_directions, 2*hidden_size]`.
                // Optional: If not specified - assumed to be 0.
                let b_default: Tensor;
                let b = match get_opt(3) {
                    Some(n) => n?,
                    None => {
                        b_default = Tensor::zeros(
                            (num_directions, 2 * hidden_size as usize),
                            DType::F32,
                            x.device(),
                        )?;
                        &b_default
                    }
                };

                // Optional tensor specifying lengths of the sequences in a batch.
                // If not specified - assumed all sequences in the batch to have length `seq_length`.
                // It has shape `[batch_size]`.
                let seq_lens_default: Tensor;
                let seq_lens = match get_opt(4) {
                    Some(n) => n?,
                    None => {
                        seq_lens_default =
                            Tensor::full(seq_length as i64, (batch_size,), x.device())?;
                        &seq_lens_default
                    }
                };
                let seq_lens_is_default =
                    (seq_lens.to_vec1::<i64>()?.iter()).all(|e| *e as usize == seq_length);
                if !seq_lens_is_default {
                    bail!("RNN currently does not support variable-length sequences. All sequences must use the full sequence length of {}", seq_length);
                }

                // Optional initial value of the hidden. If not specified - assumed to be 0.
                // It has shape `[num_directions, batch_size, hidden_size]`.
                let initial_h_default: Tensor;
                let initial_h = match get_opt(5) {
                    Some(n) => n?,
                    _ => {
                        initial_h_default = Tensor::zeros(
                            (num_directions, batch_size, hidden_size as usize),
                            DType::F32,
                            x.device(),
                        )?;
                        &initial_h_default
                    }
                };

                fn choose_activation(activation: &str, x: &Tensor) -> Result<Tensor> {
                    match activation {
                        "Tanh" => x.tanh(),
                        _ => bail!("unsupported activation {activation}"),
                    }
                }

                // these all have [num_directions, ...] shapes
                let w = w.get(0)?;
                let r = r.get(0)?;
                let b = b.get(0)?;
                let idx_wb = Tensor::arange(0, hidden_size, x.device())?;
                let idx_rb = Tensor::arange(hidden_size, 2 * hidden_size, x.device())?;
                let wb = b.index_select(&idx_wb, 0)?;
                let rb = b.index_select(&idx_rb, 0)?;
                let mut h_t = initial_h.get(0)?;
                let mut h_list: Vec<Tensor> = vec![];
                for i in 0..seq_length {
                    let xs = x.get(i)?;
                    let h = xs
                        .matmul(&w.t()?)?
                        .add(&h_t.matmul(&r.t()?)?)?
                        .add(&wb.unsqueeze(0)?)?
                        .add(&rb.unsqueeze(0)?)?;
                    let h = choose_activation(&activations[0], &h)?;
                    h_list.push(h.to_owned());
                    h_t = h;
                }
                let h = Tensor::stack(&h_list, 0)?;
                let h =
                    h.reshape((seq_length, num_directions, batch_size, hidden_size as usize))?;
                values.insert(node.output[0].clone(), h);
                values.insert(
                    node.output[1].clone(),
                    h_t.reshape((num_directions, batch_size, hidden_size as usize))?,
                );
            }
            // https://onnx.ai/onnx/operators/onnx__Xor.html
            "Xor" => {
                // Since we don't have a `DType::Bool` yet, this ensures that we are working with `0`(False) & `1`(True)
                let a = get(&node.input[0])?.gt(0_u8)?;
                let b = get(&node.input[1])?.gt(0_u8)?;

                let out = a.broadcast_add(&b)?.eq(1_u8)?;

                values.insert(node.output[0].clone(), out);
            }
            // https://onnx.ai/onnx/operators/onnx__And.html
            "And" => {
                let a = get(&node.input[0])?.gt(0_u8)?;
                let b = get(&node.input[1])?.gt(0_u8)?;

                let out = a.broadcast_mul(&b)?;

                values.insert(node.output[0].clone(), out);
            }
            // https://onnx.ai/onnx/operators/onnx__Or.html
            "Or" => {
                let a = get(&node.input[0])?.gt(0_u8)?;
                let b = get(&node.input[1])?.gt(0_u8)?;

                let out = a.broadcast_add(&b)?.gt(0_u8)?;

                values.insert(node.output[0].clone(), out);
            }
            // https://onnx.ai/onnx/operators/onnx__Sign.html
            "Sign" => {
                let input = get(&node.input[0])?;
                let output = input.sign()?;
                values.insert(node.output[0].clone(), output);
            }
            // https://onnx.ai/onnx/operators/onnx__Selu.html
            "Selu" => {
                let input = get(&node.input[0])?;
                let alpha = get_attr_opt::<f32>(node, "alpha")?
                    .copied()
                    .unwrap_or(1.6732632);
                let gamma = get_attr_opt::<f32>(node, "gamma")?
                    .copied()
                    .unwrap_or(1.050701);
                let out = candle_nn::ops::selu(input, alpha as f32, gamma as f32)?;
                values.insert(node.output[0].clone(), out);
            }
            // https://onnx.ai/onnx/operators/onnx__Elu.html
            "Elu" => {
                let input = get(&node.input[0])?;
                let alpha = get_attr_opt::<f32>(node, "alpha")?
                    .copied()
                    .unwrap_or(1.0) as f64;
                // Elu: f(x) = x if x > 0, alpha * (exp(x) - 1) if x <= 0
                // Equivalent to: max(0, x) + min(0, alpha * (exp(x) - 1))
                let zeros = input.zeros_like()?;
                let positive = input.maximum(&zeros)?;
                let negative = input.minimum(&zeros)?;
                let exp_neg = negative.exp()?;
                let neg_part = ((exp_neg - 1.0)? * alpha)?;
                let output = positive.add(&neg_part.minimum(&zeros)?)?;
                values.insert(node.output[0].clone(), output);
            }
            // https://onnx.ai/onnx/operators/onnx__Mod.html
            "Mod" => {
                let input0 = get(&node.input[0])?;
                let input1 = get(&node.input[1])?;
                let fmod = get_attr_opt::<i64>(node, "fmod")?.copied().unwrap_or(0);

                let output = if fmod == 1 {
                    // fmod behavior: a - trunc(a/b) * b
                    // For positive numbers, trunc and floor are the same
                    // For negative numbers, we need to handle differently
                    let div = input0.broadcast_div(input1)?;
                    // trunc = sign(x) * floor(abs(x))
                    let abs_div = div.abs()?;
                    let floored = abs_div.floor()?;
                    let trunc = div.sign()?.broadcast_mul(&floored)?;
                    let mul = trunc.broadcast_mul(input1)?;
                    input0.broadcast_sub(&mul)?
                } else {
                    // Default behavior: a - floor(a/b) * b
                    let div = input0.broadcast_div(input1)?;
                    let floored = div.floor()?;
                    let mul = floored.broadcast_mul(input1)?;
                    input0.broadcast_sub(&mul)?
                };

                values.insert(node.output[0].clone(), output);
            }

            // https://onnx.ai/onnx/operators/onnx__OneHot.html
            "OneHot" => {
                let indices = get(&node.input[0])?;
                let orig_shape = get(&node.input[0])?.dims().to_vec();
                let depth_tensor = get(&node.input[1])?;
                let values_tensor = get(&node.input[2])?;

                let depth = to_scalar_flexible::<i64>(depth_tensor)? as usize;
                let values_vec = values_tensor.to_vec1::<f32>()?;
                if values_vec.len() != 2 {
                    return Err(candle::Error::Msg(
                        "OneHot: expected 2-element values tensor".to_string(),
                    ));
                }
                let off_value = values_vec[0];
                let on_value = values_vec[1];

                let mut axis = node
                    .attribute
                    .iter()
                    .find(|attr| attr.name == "axis")
                    .map(|attr| attr.i)
                    .unwrap_or(-1);

                let rank = indices.rank();
                if axis < -((rank as i64) + 1) || axis > (rank as i64) {
                    return Err(candle::Error::Msg(format!(
                        "OneHot: invalid axis {axis} for rank {rank}"
                    )));
                }
                if axis < 0 {
                    axis += rank as i64 + 1;
                }

                let indices = indices.flatten_all()?;
                let indices_vec = indices.to_vec1::<i64>()?;
                let mut out = vec![off_value; depth * indices.elem_count()];
                for (i, &index) in indices_vec.iter().enumerate() {
                    let idx = if index < 0 {
                        (index + depth as i64) as usize
                    } else {
                        index as usize
                    };
                    if idx >= depth {
                        continue;
                    }
                    out[i * depth + idx] = on_value;
                }

                let mut target_shape = orig_shape;
                target_shape.push(depth);
                let output = Tensor::from_vec(out, target_shape, indices.device())?;

                let final_output = if axis as usize == output.rank() - 1 {
                    output
                } else {
                    fn move_axis_to(rank: usize, from: usize, to: usize) -> Vec<usize> {
                        let mut dims: Vec<usize> = (0..rank).collect();
                        let axis = dims.remove(from);
                        dims.insert(to, axis);
                        dims
                    }

                    let perm = move_axis_to(output.rank(), output.rank() - 1, axis as usize);
                    output.permute(&*perm)?
                };
                values.insert(node.output[0].clone(), final_output);
            }
            "HardSwish" => {
                let input = get(&node.input[0])?;
                let hard_sigmoid = candle_nn::ops::hard_sigmoid(&input)?;
                let output = input * hard_sigmoid;
                values.insert(node.output[0].clone(), output?);
            }
            "Resize" => {
                let input = get(&node.input[0])?;

                if input.rank() != 4 {
                    bail!("Unsupported rank for nearest resize: {}", input.rank());
                }

                let scales = if node.input.len() > 2 && !node.input[2].is_empty() {
                    Some(get(&node.input[2])?)
                } else {
                    None
                };

                let sizes = if node.input.len() > 3 && !node.input[3].is_empty() {
                    Some(get(&node.input[3])?)
                } else {
                    None
                };

                let output_dims = match (scales, sizes) {
                    (Some(_), Some(_)) => {
                        bail!("Scales and sizes cannot both be set for Resize operation")
                    }
                    (Some(scales_tensor), None) => {
                        let scale_values = scales_tensor.to_vec1::<f32>()?;
                        input
                            .dims()
                            .iter()
                            .enumerate()
                            .map(|(i, &d)| (d as f32 * scale_values[i]) as usize)
                            .collect::<Vec<_>>()
                    }
                    (None, Some(sizes_tensor)) => sizes_tensor
                        .to_vec1::<i64>()?
                        .iter()
                        .map(|&d| d as usize)
                        .collect::<Vec<_>>(),
                    (None, None) => bail!("Either scales or sizes should be present"),
                };

                let coordinate_transformation_mode =
                    get_attr_opt::<str>(node, "coordinate_transformation_mode")?
                        .unwrap_or("half_pixel");
                // Interpolation mode: nearest, linear, or cubic.
                let mode = get_attr_opt::<str>(node, "mode")?.unwrap_or("nearest");
                // How to determine the "nearest" pixel in nearest interpolation mode.
                let nearest_mode =
                    get_attr_opt::<str>(node, "nearest_mode")?.unwrap_or("round_prefer_floor");

                if mode != "nearest" {
                    bail!("Unsupported resize mode: {}", mode);
                }

                if nearest_mode != "floor" {
                    bail!("Unsupported nearest_mode for resize: {}", nearest_mode);
                }

                if coordinate_transformation_mode != "asymmetric" {
                    bail!(
                        "Unsupported coordinate_transformation_mode for resize: {}",
                        coordinate_transformation_mode
                    );
                }

                let h = output_dims[2];
                let w = output_dims[3];
                let output = input.upsample_nearest2d(h, w)?;

                values.insert(node.output[0].clone(), output);
            }
            "Trilu" => {
                let input = get(&node.input[0])?;

                // Get the diagonal offset 'k' from the second input if provided
                let k = if node.input.len() > 1 && !node.input[1].is_empty() {
                    to_vec0_flexible::<i64>(get(&node.input[1])?)?
                } else {
                    0
                };

                // Get the 'upper' attribute
                let upper = get_attr_opt::<i64>(node, "upper")?.copied().unwrap_or(1);

                // For batched inputs, we need to handle each matrix separately
                let dims = input.dims();
                if dims.len() < 2 {
                    bail!("Trilu expects input with at least 2 dimensions: {:?}", dims);
                }

                // Get the last two dimensions which represent the matrix
                let n = dims[dims.len() - 2];
                let m = dims[dims.len() - 1];
                let max_dim = std::cmp::max(n, m);

                // Handle the diagonal offset k
                let mask = if k != 0 {
                    let mut data = vec![0u32; n * m];
                    for i in 0..n {
                        for j in 0..m {
                            if (upper != 0 && (j as i64) >= (i as i64) + k)
                                || (upper == 0 && (j as i64) <= (i as i64) + k)
                            {
                                data[i * m + j] = 1u32;
                            }
                        }
                    }
                    Tensor::from_vec(data, (n, m), input.device())?.to_dtype(input.dtype())?
                } else if upper == 0 {
                    Tensor::tril2(max_dim, input.dtype(), input.device())?
                } else {
                    Tensor::triu2(max_dim, input.dtype(), input.device())?
                };

                let final_mask = if n != m {
                    mask.narrow(0, 0, n)?.narrow(1, 0, m)?
                } else {
                    mask
                };

                let output = (input * &final_mask)?;

                values.insert(node.output[0].clone(), output);
            }
            "ScatterND" => {
                let data = get(&node.input[0])?;

                let indices = get(&node.input[1])?;
                let indices = indices.to_dtype(DType::I64)?;

                let updates = get(&node.input[2])?;

                let reduction = get_attr_opt::<str>(node, "reduction")?.unwrap_or("none");

                let indices_shape = indices.dims();
                let data_shape = data.dims();
                let _updates_shape = updates.dims();

                // Last dimension of indices represents the depth of indexing
                let k = indices_shape.last().unwrap().clone();

                if k > data.rank() {
                    bail!("ScatterND expects k (indices.shape[-1]) to be at most the rank of data");
                }

                let num_updates = indices_shape[..indices_shape.len() - 1]
                    .iter()
                    .product::<usize>();

                let flat_indices = if indices.rank() == 1 && k == 1 {
                    indices.unsqueeze(0)?
                } else {
                    indices.reshape((num_updates, k))?
                };

                // Calculate the shape of each update element
                let update_element_shape = if k < data_shape.len() {
                    data_shape[k..].to_vec()
                } else {
                    vec![]
                };

                // Expected shape for updates based on indices and target tensor
                let expected_updates_shape = {
                    let mut shape = indices_shape[..indices_shape.len() - 1].to_vec();
                    shape.extend(&update_element_shape);
                    shape
                };

                // Validate or reshape updates to expected shape
                let updates = if updates.dims() != expected_updates_shape {
                    if updates.rank() == 0 {
                        // Handle scalar updates
                        let mut target_shape = vec![num_updates];
                        target_shape.extend(&update_element_shape);
                        updates.broadcast_as(target_shape)?
                    } else {
                        // Try to broadcast or reshape updates to expected shape
                        let flat_shape =
                            vec![num_updates, update_element_shape.iter().product::<usize>()];
                        let flattened = updates.reshape(flat_shape)?;
                        flattened.reshape(expected_updates_shape)?
                    }
                } else {
                    updates.clone()
                };

                let mut output = data.clone();

                // convert indices to flat indices
                let mut flat_output = output.flatten_all()?;
                let flat_updates = if update_element_shape.is_empty() {
                    updates.reshape(num_updates)?
                } else {
                    let product = update_element_shape.iter().product::<usize>();
                    updates.reshape((num_updates, product))?
                };

                // Calculate strides for the output tensor
                let mut strides: Vec<usize> = vec![1];
                for i in (0..data_shape.len() - 1).rev() {
                    strides.push(strides.last().unwrap() * data_shape[i + 1]);
                }
                strides.reverse();

                // Process each update
                for i in 0..num_updates {
                    let index_slice = flat_indices.narrow(0, i, 1)?;
                    let indices_vec = index_slice.squeeze(0)?.to_vec1::<i64>()?;

                    // Convert multi-dimensional indices to flat index
                    let mut flat_idx: usize = 0;
                    for (dim, &idx) in indices_vec.iter().enumerate() {
                        let dim_size = data_shape[dim] as i64;
                        let norm_idx = if idx < 0 { dim_size + idx } else { idx };

                        if norm_idx < 0 || norm_idx >= dim_size {
                            bail!(
                                "Index {} out of bounds for dimension {} with size {}",
                                idx,
                                dim,
                                dim_size
                            );
                        }

                        flat_idx += (norm_idx as usize) * strides[dim];
                    }

                    // Extract current update
                    let update_slice = if update_element_shape.is_empty() {
                        flat_updates.narrow(0, i, 1)?.squeeze(0)?
                    } else {
                        flat_updates.narrow(0, i, 1)?
                    };

                    match reduction {
                        "add" => {
                            if update_element_shape.is_empty() {
                                let existing = flat_output.narrow(0, flat_idx, 1)?;
                                let new_value = existing.add(&update_slice.unsqueeze(0)?)?;
                                flat_output = flat_output.slice_scatter(&new_value, 0, flat_idx)?;
                            } else {
                                let slice_size = update_element_shape.iter().product::<usize>();
                                let existing = flat_output.narrow(0, flat_idx, slice_size)?;
                                let new_value = existing.add(&update_slice)?;
                                flat_output = flat_output.slice_scatter(&new_value, 0, flat_idx)?;
                            }
                        }
                        "none" | _ => {
                            if update_element_shape.is_empty() {
                                flat_output = flat_output.slice_scatter(
                                    &update_slice.unsqueeze(0)?,
                                    0,
                                    flat_idx,
                                )?;
                            } else {
                                flat_output =
                                    flat_output.slice_scatter(&update_slice, 0, flat_idx)?;
                            }
                        }
                    }
                }

                // Reshape flat output back to original shape
                output = flat_output.reshape(data_shape.to_vec())?;

                values.insert(node.output[0].clone(), output);
            }
            // CosyVoice3 GRU operator support
            "GRU" => {
                // === 1. Parse attributes ===
                let direction = get_attr_opt::<str>(node, "direction")?
                    .unwrap_or("forward");
                if direction != "forward" {
                    bail!("GRU currently only supports direction == \"forward\"");
                }
                let num_directions = 1usize;
                let hidden_size: i64 = *get_attr(node, "hidden_size")?;
                let hidden_size = hidden_size as usize;

                // linear_before_reset: 0 = r_t * (H_prev * R + Rb), 1 = (r_t * H_prev) * R + Rb
                let linear_before_reset = get_attr_opt::<i64>(node, "linear_before_reset")?
                    .copied()
                    .unwrap_or(0);
                if linear_before_reset != 0 {
                    bail!("GRU currently only supports linear_before_reset == 0");
                }

                let layout = get_attr_opt::<i64>(node, "layout")?
                    .copied()
                    .unwrap_or(0);
                if layout != 0 {
                    bail!("GRU currently only supports layout == 0");
                }

                // === 2. Get input tensors ===
                // X: [seq_length, batch_size, input_size]
                let x = get(&node.input[0])?;
                let (seq_length, batch_size, input_size) = x.dims3()?;

                // W: [num_directions, 3*hidden_size, input_size] - input weights
                let w = get(&node.input[1])?;

                // R: [num_directions, 3*hidden_size, hidden_size] - recurrent weights
                let r = get(&node.input[2])?;

                // B: [num_directions, 6*hidden_size] - optional bias
                let b = if node.input.len() > 3 && !node.input[3].is_empty() {
                    get(&node.input[3])?.clone()
                } else {
                    Tensor::zeros(
                        (num_directions, 6 * hidden_size),
                        x.dtype(),
                        x.device(),
                    )?
                };

                // initial_h: [num_directions, batch_size, hidden_size] - optional initial hidden state
                let initial_h = if node.input.len() > 5 && !node.input[5].is_empty() {
                    get(&node.input[5])?.clone()
                } else {
                    Tensor::zeros(
                        (num_directions, batch_size, hidden_size),
                        x.dtype(),
                        x.device(),
                    )?
                };

                // === 3. Extract single direction weights (assuming forward) ===
                let w = w.get(0)?; // [3*hidden_size, input_size]
                let r = r.get(0)?; // [3*hidden_size, hidden_size]
                let b = b.get(0)?; // [6*hidden_size]
                let h = initial_h.get(0)?; // [batch_size, hidden_size]

                // === 4. Separate biases: Wb and Rb ===
                let wb = b.narrow(0, 0, 3 * hidden_size)?;
                let rb = b.narrow(0, 3 * hidden_size, 3 * hidden_size)?;

                // === 5. Reorder weights: ONNX (zrh) -> candle-nn (rzn) ===
                // ONNX order: z(update), r(reset), h(hidden)
                // candle-nn order: r(reset), z(update), n(hidden)
                let w_z = w.narrow(0, 0, hidden_size)?;
                let w_r = w.narrow(0, hidden_size, hidden_size)?;
                let w_h = w.narrow(0, 2 * hidden_size, hidden_size)?;
                let w = Tensor::cat(&[&w_r, &w_z, &w_h], 0)?; // [3*hidden_size, input_size]

                let r_z = r.narrow(0, 0, hidden_size)?;
                let r_r = r.narrow(0, hidden_size, hidden_size)?;
                let r_h = r.narrow(0, 2 * hidden_size, hidden_size)?;
                let r = Tensor::cat(&[&r_r, &r_z, &r_h], 0)?; // [3*hidden_size, hidden_size]

                let wb_z = wb.narrow(0, 0, hidden_size)?;
                let wb_r = wb.narrow(0, hidden_size, hidden_size)?;
                let wb_h = wb.narrow(0, 2 * hidden_size, hidden_size)?;
                let wb = Tensor::cat(&[&wb_r, &wb_z, &wb_h], 0)?;

                let rb_z = rb.narrow(0, 0, hidden_size)?;
                let rb_r = rb.narrow(0, hidden_size, hidden_size)?;
                let rb_h = rb.narrow(0, 2 * hidden_size, hidden_size)?;
                let rb = Tensor::cat(&[&rb_r, &rb_z, &rb_h], 0)?;

                // === 6. Build candle-nn GRU ===
                let vmap = candle_nn::VarMap::new();
                {
                    let mut data = vmap.data().lock().unwrap();
                    data.insert("weight_ih_l0".to_string(), candle::Var::from_tensor(&w)?);
                    data.insert("weight_hh_l0".to_string(), candle::Var::from_tensor(&r)?);
                    data.insert("bias_ih_l0".to_string(), candle::Var::from_tensor(&wb)?);
                    data.insert("bias_hh_l0".to_string(), candle::Var::from_tensor(&rb)?);
                }

                use candle_nn::rnn::RNN as _;
                let gru = candle_nn::rnn::gru(
                    input_size,
                    hidden_size,
                    candle_nn::rnn::GRUConfig::default(),
                    candle_nn::VarBuilder::from_varmap(&vmap, w.dtype(), w.device()),
                )?;

                // === 7. Execute GRU sequence ===
                let mut gru_state = candle_nn::rnn::GRUState { h };
                let mut h_outputs: Vec<Tensor> = Vec::new();

                for t in 0..seq_length {
                    let x_t = x.get(t)?; // [batch_size, input_size]
                    gru_state = gru.step(&x_t, &gru_state)?;
                    h_outputs.push(gru_state.h.clone());
                }

                // === 8. Build outputs ===
                // Y: [seq_length, num_directions, batch_size, hidden_size]
                if !node.output.is_empty() && !node.output[0].is_empty() {
                    let y = Tensor::stack(&h_outputs, 0)?; // [seq_length, batch_size, hidden_size]
                    let y = y.unsqueeze(1)?; // [seq_length, 1, batch_size, hidden_size]
                    values.insert(node.output[0].clone(), y);
                }

                // Y_h: [num_directions, batch_size, hidden_size]
                if node.output.len() > 1 && !node.output[1].is_empty() {
                    let y_h = gru_state.h.unsqueeze(0)?; // [1, batch_size, hidden_size]
                    values.insert(node.output[1].clone(), y_h);
                }
            }
            op_type => bail!("unsupported op_type {op_type} for op {node:?}"),
        }
    }
    graph
        .output
        .iter()
        .map(|output| match values.remove(&output.name) {
            None => bail!("cannot find output {}", output.name),
            Some(value) => Ok((output.name.clone(), value)),
        })
        .collect()
}

fn broadcast_shape(shape_a: &[usize], shape_b: &[usize]) -> Result<Vec<usize>> {
    let (longest, shortest) = if shape_a.len() > shape_b.len() {
        (shape_a, shape_b)
    } else {
        (shape_b, shape_a)
    };
    let diff = longest.len() - shortest.len();
    let mut target_shape = longest[0..diff].to_vec();
    for (dim1, dim2) in longest[diff..].iter().zip(shortest.iter()) {
        if *dim1 == *dim2 || *dim2 == 1 || *dim1 == 1 {
            target_shape.push(usize::max(*dim1, *dim2));
        } else {
            bail!(
                "Expand: incompatible shapes for broadcast, {:?} and {:?}",
                shape_a,
                shape_b
            );
        }
    }
    Ok(target_shape)
}

fn broadcast_shape_from_many(shapes: &[&[usize]]) -> Result<Vec<usize>> {
    if shapes.is_empty() {
        return Ok(Vec::new());
    }
    let mut shape_out = shapes[0].to_vec();
    for shape in shapes[1..].iter() {
        shape_out = broadcast_shape(&shape_out, shape)?;
    }
    Ok(shape_out)
}

/// Extract scalar from tensors that may be wrapped in extra dimensions.
/// Some ONNX exports use shape [1] or [1,1] where scalars are expected.
/// Only accepts single-element tensors; multi-element tensors still fail.
fn to_scalar_flexible<T: candle::WithDType>(t: &Tensor) -> Result<T> {
    if t.rank() > 0 && t.elem_count() == 1 {
        t.flatten_all()?.i(0)?.to_scalar::<T>()
    } else {
        t.to_scalar::<T>()
    }
}

/// Same as to_scalar_flexible but returns via to_vec0 for types that need it.
fn to_vec0_flexible<T: candle::WithDType>(t: &Tensor) -> Result<T> {
    if t.rank() > 0 && t.elem_count() == 1 {
        t.flatten_all()?.i(0)?.to_vec0::<T>()
    } else {
        t.to_vec0::<T>()
    }
}
