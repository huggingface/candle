#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use candle::{Device, Result, Tensor};
use candle_onnx::onnx::{GraphProto, ModelProto, NodeProto, ValueInfoProto};
use std::collections::HashMap;

const INPUT_X: &str = "x";
const INPUT_Y: &str = "y";
const OUTPUT_Z: &str = "z";

fn create_model_proto_with_graph(graph: Option<GraphProto>) -> ModelProto {
    ModelProto {
        metadata_props: vec![],
        training_info: vec![],
        functions: vec![],
        ir_version: 0,
        opset_import: vec![],
        producer_name: "".to_string(),
        producer_version: "".to_string(),
        domain: "".to_string(),
        model_version: 0,
        doc_string: "".to_string(),
        graph,
    }
}

#[test]
fn test_evaluation_fails_without_defined_graph() -> Result<()> {
    let manual_graph = create_model_proto_with_graph(None);

    let inputs: HashMap<String, Tensor> = HashMap::new();

    match candle_onnx::simple_eval(&manual_graph, inputs) {
        Err(err) => assert_eq!(err.to_string(), "no graph defined in proto"),
        Ok(_) => panic!("Expected an error due to undefined graph"),
    }

    Ok(())
}

// "Add"
#[test]
fn test_add_operation() -> Result<()> {
    let manual_graph = create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: "Add".to_string(),
            domain: "".to_string(),
            attribute: vec![],
            input: vec!["x".to_string(), "y".to_string()],
            output: vec!["z".to_string()],
            name: "".to_string(),
            doc_string: "".to_string(),
        }],
        name: "".to_string(),
        initializer: vec![],
        input: vec![],
        output: vec![ValueInfoProto {
            name: "z".to_string(),
            doc_string: "".to_string(),
            r#type: None,
        }],
        value_info: vec![],
        doc_string: "".to_string(),
        sparse_initializer: vec![],
        quantization_annotation: vec![],
    }));

    let mut inputs: HashMap<String, Tensor> = HashMap::new();
    inputs.insert(INPUT_X.to_string(), Tensor::new(&[2.], &Device::Cpu)?);
    inputs.insert(INPUT_Y.to_string(), Tensor::new(&[2.], &Device::Cpu)?);

    let eval = candle_onnx::simple_eval(&manual_graph, inputs)?;
    assert_eq!(eval.len(), 1);

    let z = eval.get(OUTPUT_Z).expect("Output 'z' not found");
    let first = z
        .to_vec1::<f64>()?
        .to_vec()
        .get(0)
        .expect("Failed to get first element")
        .clone();
    assert_eq!(first, 4.0f64);

    Ok(())
}

// "Sub"
#[test]
fn test_sub_operation() -> Result<()> {
    let manual_graph = create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: "Sub".to_string(),
            domain: "".to_string(),
            attribute: vec![],
            input: vec!["x".to_string(), "y".to_string()],
            output: vec!["z".to_string()],
            name: "".to_string(),
            doc_string: "".to_string(),
        }],
        name: "".to_string(),
        initializer: vec![],
        input: vec![],
        output: vec![ValueInfoProto {
            name: "z".to_string(),
            doc_string: "".to_string(),
            r#type: None,
        }],
        value_info: vec![],
        doc_string: "".to_string(),
        sparse_initializer: vec![],
        quantization_annotation: vec![],
    }));

    let mut inputs: HashMap<String, Tensor> = HashMap::new();
    inputs.insert(INPUT_X.to_string(), Tensor::new(&[2.], &Device::Cpu)?);
    inputs.insert(INPUT_Y.to_string(), Tensor::new(&[2.], &Device::Cpu)?);

    let eval = candle_onnx::simple_eval(&manual_graph, inputs)?;
    assert_eq!(eval.len(), 1);

    let z = eval.get(OUTPUT_Z).expect("Output 'z' not found");
    let first = z
        .to_vec1::<f64>()?
        .to_vec()
        .get(0)
        .expect("Failed to get first element")
        .clone();
    assert_eq!(first, 0.0f64);

    Ok(())
}

// "Mul"
#[test]
fn test_mul_operation() -> Result<()> {
    let manual_graph = create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: "Mul".to_string(),
            domain: "".to_string(),
            attribute: vec![],
            input: vec!["x".to_string(), "y".to_string()],
            output: vec!["z".to_string()],
            name: "".to_string(),
            doc_string: "".to_string(),
        }],
        name: "".to_string(),
        initializer: vec![],
        input: vec![],
        output: vec![ValueInfoProto {
            name: "z".to_string(),
            doc_string: "".to_string(),
            r#type: None,
        }],
        value_info: vec![],
        doc_string: "".to_string(),
        sparse_initializer: vec![],
        quantization_annotation: vec![],
    }));

    let mut inputs: HashMap<String, Tensor> = HashMap::new();
    inputs.insert(INPUT_X.to_string(), Tensor::new(&[2.], &Device::Cpu)?);
    inputs.insert(INPUT_Y.to_string(), Tensor::new(&[2.], &Device::Cpu)?);

    let eval = candle_onnx::simple_eval(&manual_graph, inputs)?;
    assert_eq!(eval.len(), 1);

    let z = eval.get(OUTPUT_Z).expect("Output 'z' not found");
    let first = z
        .to_vec1::<f64>()?
        .to_vec()
        .get(0)
        .expect("Failed to get first element")
        .clone();
    assert_eq!(first, 4.0f64);

    Ok(())
}

// "Div"
#[test]
fn test_div_operation() -> Result<()> {
    let manual_graph = create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: "Div".to_string(),
            domain: "".to_string(),
            attribute: vec![],
            input: vec!["x".to_string(), "y".to_string()],
            output: vec!["z".to_string()],
            name: "".to_string(),
            doc_string: "".to_string(),
        }],
        name: "".to_string(),
        initializer: vec![],
        input: vec![],
        output: vec![ValueInfoProto {
            name: "z".to_string(),
            doc_string: "".to_string(),
            r#type: None,
        }],
        value_info: vec![],
        doc_string: "".to_string(),
        sparse_initializer: vec![],
        quantization_annotation: vec![],
    }));

    let mut inputs: HashMap<String, Tensor> = HashMap::new();
    inputs.insert(INPUT_X.to_string(), Tensor::new(&[2.], &Device::Cpu)?);
    inputs.insert(INPUT_Y.to_string(), Tensor::new(&[2.], &Device::Cpu)?);

    let eval = candle_onnx::simple_eval(&manual_graph, inputs)?;
    assert_eq!(eval.len(), 1);

    let z = eval.get(OUTPUT_Z).expect("Output 'z' not found");
    let first = z
        .to_vec1::<f64>()?
        .to_vec()
        .get(0)
        .expect("Failed to get first element")
        .clone();

    assert_eq!(first, 1.0f64);

    Ok(())
}

// "Equal"
#[test]
fn test_equal_operation() -> Result<()> {
    let manual_graph = create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: "Equal".to_string(),
            domain: "".to_string(),
            attribute: vec![],
            input: vec!["x".to_string(), "y".to_string()],
            output: vec!["z".to_string()],
            name: "".to_string(),
            doc_string: "".to_string(),
        }],
        name: "".to_string(),
        initializer: vec![],
        input: vec![],
        output: vec![ValueInfoProto {
            name: "z".to_string(),
            doc_string: "".to_string(),
            r#type: None,
        }],
        value_info: vec![],
        doc_string: "".to_string(),
        sparse_initializer: vec![],
        quantization_annotation: vec![],
    }));

    let mut inputs: HashMap<String, Tensor> = HashMap::new();
    inputs.insert(INPUT_X.to_string(), Tensor::new(&[2.], &Device::Cpu)?);
    inputs.insert(INPUT_Y.to_string(), Tensor::new(&[2.], &Device::Cpu)?);

    let eval = candle_onnx::simple_eval(&manual_graph, inputs)?;
    assert_eq!(eval.len(), 1);

    let z = eval.get(OUTPUT_Z).expect("Output 'z' not found");
    let first = z.to_dtype(candle::DType::U8)?.to_vec1::<u8>()?.to_vec()[0];
    assert_eq!(first, 1);

    Ok(())
}

// Below are ops that are implemented but not tested yet

// "Not"
// #[test]

// "MatMul"
// #[test]

// "Reshape"
// #[test]

// "LogSoftmax"
// #[test]

// "Softmax"
// #[test]

// "Transpose"
// #[test]

// "Dropout"
// #[test]

// "MaxPool"
// #[test]

// "AveragePool"
// #[test]

// "BatchNormalization"
// #[test]

// "Squeeze"
// #[test]

// "ConstantOfShape"
// #[test]

// "Unsqueeze"
// #[test]

// "Clip"
// #[test]

// "Gather"
// #[test]

// "Shape"
// #[test]

// "Conv"
// #[test]

// "Concat"
// #[test]

// "Abs"
// #[test]

// "Cos"
// #[test]

// "Sin"
// #[test]

// "Neg"
// #[test]

// "Erf"
// #[test]

// "Tanh"
// #[test]

// "Sigmoid"
// #[test]

// "Gelu"
// #[test]

// "Relu"
// #[test]

// "Constant"
// #[test]

// "Cast"
// #[test]
