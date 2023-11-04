use crate::onnx;
use candle::{Result, Tensor};
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
        None => candle::bail!("no graph defined in proto"),
        Some(graph) => graph,
    };
    // TODO: validate the inputs.
    let mut values = inputs;
    // The nodes are topologically sorted so we can just process them in order.
    for node in graph.node.iter() {
        let get = |input_name: &str| match values.get(input_name) {
            Some(value) => Ok(value),
            None => candle::bail!("cannot find {input_name} for op {}", node.name),
        };
        match node.op_type.as_str() {
            "Relu" => {
                // TODO: Validate node.input.
                let input = get(&node.input[0])?;
                let output = input.relu()?;
                values.insert(node.output[0].clone(), output);
            }
            op_type => candle::bail!("unsupported op_type {op_type} for op {}", node.name),
        }
    }
    graph
        .output
        .iter()
        .map(|output| match values.remove(&output.name) {
            None => candle::bail!("cannot find output {}", output.name),
            Some(value) => Ok((output.name.clone(), value)),
        })
        .collect()
}
