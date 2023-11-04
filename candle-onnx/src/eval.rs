use crate::onnx;
use candle::{bail, Result, Tensor};
use std::collections::HashMap;

pub type Value = Tensor;

// This function provides a direct evaluation of the proto.
// Longer-term, we should first convert the proto to an intermediate representation of the compute
// graph so as to make multiple evaluations more efficient.
// An example upside of this would be to remove intermediary values when they are not needed
// anymore.
pub fn simple_eval(
    model: &onnx::ModelProto,
    inputs: HashMap<String, Value>,
) -> Result<HashMap<String, Value>> {
    let graph = match &model.graph {
        None => bail!("no graph defined in proto"),
        Some(graph) => graph,
    };
    // TODO: validate the inputs.
    let mut values = inputs;
    // The nodes are topologically sorted so we can just process them in order.
    for node in graph.node.iter() {
        let get = |input_name: &str| match values.get(input_name) {
            Some(value) => Ok(value),
            None => bail!("cannot find {input_name} for op {}", node.name),
        };
        // TODO: Validate node.input for each operator.
        match node.op_type.as_str() {
            "Add" => {
                let input0 = get(&node.input[0])?;
                let input1 = get(&node.input[0])?;
                let output = input0.broadcast_add(input1)?;
                values.insert(node.output[0].clone(), output);
            }
            "Sub" => {
                let input0 = get(&node.input[0])?;
                let input1 = get(&node.input[0])?;
                let output = input0.broadcast_sub(input1)?;
                values.insert(node.output[0].clone(), output);
            }
            "Mul" => {
                let input0 = get(&node.input[0])?;
                let input1 = get(&node.input[0])?;
                let output = input0.broadcast_mul(input1)?;
                values.insert(node.output[0].clone(), output);
            }
            "Div" => {
                let input0 = get(&node.input[0])?;
                let input1 = get(&node.input[0])?;
                let output = input0.broadcast_div(input1)?;
                values.insert(node.output[0].clone(), output);
            }
            "MatMul" => {
                let input0 = get(&node.input[0])?;
                let input1 = get(&node.input[0])?;
                let output = input0.broadcast_matmul(input1)?;
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
            "Cast" => {
                let input = get(&node.input[0])?;
                let dtype = match node.attribute.iter().find(|attr| attr.name == "to") {
                    None => {
                        bail!("cannot find the 'to' attribute in 'Cast' for {}", node.name)
                    }
                    Some(dtype) => match dtype.r#type() {
                        crate::onnx::attribute_proto::AttributeType::Floats => candle::DType::F32,
                        rtype => bail!("unsupported 'to' type {rtype:?} for {}", node.name),
                    },
                };
                let output = input.to_dtype(dtype)?;
                values.insert(node.output[0].clone(), output);
            }
            op_type => bail!("unsupported op_type {op_type} for op {}", node.name),
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
