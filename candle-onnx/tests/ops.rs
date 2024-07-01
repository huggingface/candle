#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use candle::test_utils::to_vec2_round;
use candle::{DType, Device, NdArray, Result, Tensor};
use candle_onnx::onnx::attribute_proto::AttributeType;
use candle_onnx::onnx::tensor_proto::DataType;
use candle_onnx::onnx::tensor_shape_proto::{dimension, Dimension};
use candle_onnx::onnx::{type_proto, TensorProto, TensorShapeProto, TypeProto};
use candle_onnx::onnx::{AttributeProto, GraphProto, ModelProto, NodeProto, ValueInfoProto};
use std::collections::HashMap;

const INPUT_X: &str = "x";
const INPUT_Y: &str = "y";
const INPUT_A: &str = "a";
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
            input: vec![INPUT_X.to_string(), INPUT_Y.to_string()],
            output: vec![OUTPUT_Z.to_string()],
            name: "".to_string(),
            doc_string: "".to_string(),
        }],
        name: "".to_string(),
        initializer: vec![],
        input: vec![],
        output: vec![ValueInfoProto {
            name: OUTPUT_Z.to_string(),
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
    let first = z.to_vec1::<f64>()?[0];
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
            input: vec![INPUT_X.to_string(), INPUT_Y.to_string()],
            output: vec![OUTPUT_Z.to_string()],
            name: "".to_string(),
            doc_string: "".to_string(),
        }],
        name: "".to_string(),
        initializer: vec![],
        input: vec![],
        output: vec![ValueInfoProto {
            name: OUTPUT_Z.to_string(),
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
    let first = z.to_vec1::<f64>()?[0];
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
            input: vec![INPUT_X.to_string(), INPUT_Y.to_string()],
            output: vec![OUTPUT_Z.to_string()],
            name: "".to_string(),
            doc_string: "".to_string(),
        }],
        name: "".to_string(),
        initializer: vec![],
        input: vec![],
        output: vec![ValueInfoProto {
            name: OUTPUT_Z.to_string(),
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
    let first = z.to_vec1::<f64>()?[0];
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
            input: vec![INPUT_X.to_string(), INPUT_Y.to_string()],
            output: vec![OUTPUT_Z.to_string()],
            name: "".to_string(),
            doc_string: "".to_string(),
        }],
        name: "".to_string(),
        initializer: vec![],
        input: vec![],
        output: vec![ValueInfoProto {
            name: OUTPUT_Z.to_string(),
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
    let first = z.to_vec1::<f64>()?[0];
    assert_eq!(first, 1.0f64);
    Ok(())
}

// "Exp"
#[test]
fn test_exp_operation() -> Result<()> {
    let manual_graph = create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: "Exp".to_string(),
            domain: "".to_string(),
            attribute: vec![],
            input: vec![INPUT_X.to_string()],
            output: vec![OUTPUT_Z.to_string()],
            name: "".to_string(),
            doc_string: "".to_string(),
        }],
        name: "".to_string(),
        initializer: vec![],
        input: vec![],
        output: vec![ValueInfoProto {
            name: OUTPUT_Z.to_string(),
            doc_string: "".to_string(),
            r#type: None,
        }],
        value_info: vec![],
        doc_string: "".to_string(),
        sparse_initializer: vec![],
        quantization_annotation: vec![],
    }));

    let x = Tensor::from_vec(vec![-1.0f32, 0.0f32, 1.0f32, 2.0f32], &[2, 2], &Device::Cpu)?;

    let mut inputs: HashMap<String, Tensor> = HashMap::new();
    inputs.insert(INPUT_X.to_string(), x);

    let eval = candle_onnx::simple_eval(&manual_graph, inputs)?;
    assert_eq!(eval.len(), 1);

    let z = eval.get(OUTPUT_Z).expect("Output 'z' not found");

    let results = z.to_vec2::<f32>()?;

    assert_eq!(results[0][0], 0.36787944f32);
    assert_eq!(results[0][1], 1.0f32);
    assert_eq!(results[1], vec![std::f32::consts::E, 7.389056f32]);

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
            input: vec![INPUT_X.to_string(), INPUT_Y.to_string()],
            output: vec![OUTPUT_Z.to_string()],
            name: "".to_string(),
            doc_string: "".to_string(),
        }],
        name: "".to_string(),
        initializer: vec![],
        input: vec![],
        output: vec![ValueInfoProto {
            name: OUTPUT_Z.to_string(),
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

// "Not"
#[test]
fn test_not_operation() -> Result<()> {
    let manual_graph = create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: "Not".to_string(),
            domain: "".to_string(),
            attribute: vec![],
            input: vec![INPUT_X.to_string()],
            output: vec![OUTPUT_Z.to_string()],
            name: "".to_string(),
            doc_string: "".to_string(),
        }],
        name: "".to_string(),
        initializer: vec![],
        input: vec![],
        output: vec![ValueInfoProto {
            name: OUTPUT_Z.to_string(),
            doc_string: "".to_string(),
            r#type: None,
        }],
        value_info: vec![],
        doc_string: "".to_string(),
        sparse_initializer: vec![],
        quantization_annotation: vec![],
    }));

    let mut inputs: HashMap<String, Tensor> = HashMap::new();
    inputs.insert(INPUT_X.to_string(), Tensor::new(&[0.], &Device::Cpu)?);

    let eval = candle_onnx::simple_eval(&manual_graph, inputs)?;
    assert_eq!(eval.len(), 1);

    let z = eval.get(OUTPUT_Z).expect("Output 'z' not found");
    let first = z.to_dtype(candle::DType::U8)?.to_vec1::<u8>()?.to_vec()[0];
    assert_eq!(first, 1);

    Ok(())
}

// "MatMul"
#[test]
fn test_matmul_operation() -> Result<()> {
    let manual_graph = create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: "MatMul".to_string(),
            domain: "".to_string(),
            attribute: vec![],
            input: vec![INPUT_X.to_string(), INPUT_Y.to_string()],
            output: vec![OUTPUT_Z.to_string()],
            name: "".to_string(),
            doc_string: "".to_string(),
        }],
        name: "".to_string(),
        initializer: vec![],
        input: vec![],
        output: vec![ValueInfoProto {
            name: OUTPUT_Z.to_string(),
            doc_string: "".to_string(),
            r#type: None,
        }],
        value_info: vec![],
        doc_string: "".to_string(),
        sparse_initializer: vec![],
        quantization_annotation: vec![],
    }));

    let mut inputs: HashMap<String, Tensor> = HashMap::new();
    inputs.insert(
        INPUT_X.to_string(),
        Tensor::from_vec(
            //
            vec![1.0f32, 2.0f32, 3.0f32, 4.0f32],
            &[2, 2],
            &Device::Cpu,
        )?,
    );
    inputs.insert(
        INPUT_Y.to_string(),
        Tensor::from_vec(
            //
            vec![5.0f32, 6.0f32, 7.0f32, 8.0f32],
            &[2, 2],
            &Device::Cpu,
        )?,
    );

    let eval = candle_onnx::simple_eval(&manual_graph, inputs)?;
    assert_eq!(eval.len(), 1);

    let z = eval.get(OUTPUT_Z).expect("Output 'z' not found");
    let results = z.to_vec2::<f32>()?;
    assert_eq!(results, vec![vec![19.0, 22.0], vec![43.0, 50.0]]);

    Ok(())
}

// "Reshape"
#[test]
fn test_reshape_operation() -> Result<()> {
    let manual_graph = create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: "Reshape".to_string(),
            domain: "".to_string(),
            attribute: vec![],
            input: vec![INPUT_X.to_string(), INPUT_Y.to_string()],
            output: vec![OUTPUT_Z.to_string()],
            name: "".to_string(),
            doc_string: "".to_string(),
        }],
        name: "".to_string(),
        initializer: vec![],
        input: vec![
            ValueInfoProto {
                name: INPUT_X.to_string(),
                doc_string: "".to_string(),
                r#type: None,
            },
            ValueInfoProto {
                name: INPUT_Y.to_string(),
                doc_string: "".to_string(),
                r#type: None,
            },
        ],
        output: vec![ValueInfoProto {
            name: OUTPUT_Z.to_string(),
            doc_string: "".to_string(),
            r#type: None,
        }],
        value_info: vec![],
        doc_string: "".to_string(),
        sparse_initializer: vec![],
        quantization_annotation: vec![],
    }));

    let x = Tensor::from_vec(
        //
        vec![1.0f32, 2.0f32, 3.0f32, 4.0f32],
        &[2, 2],
        &Device::Cpu,
    )?;
    let y = Tensor::from_vec(
        //
        vec![4i64],
        &[1],
        &Device::Cpu,
    )?;

    let mut inputs: HashMap<String, Tensor> = HashMap::new();
    inputs.insert(INPUT_X.to_string(), x);
    inputs.insert(INPUT_Y.to_string(), y);

    let eval = candle_onnx::simple_eval(&manual_graph, inputs)?;
    assert_eq!(eval.len(), 1);

    let z = eval.get(OUTPUT_Z).expect("Output 'z' not found");

    let results = z.to_vec1::<f32>()?;

    assert_eq!(results, vec![1.0, 2.0, 3.0, 4.0]);

    Ok(())
}

// "LogSoftmax"
#[test]
fn test_logsoftmax_operation() -> Result<()> {
    let manual_graph = create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: "LogSoftmax".to_string(),
            domain: "".to_string(),
            attribute: vec![],
            input: vec![INPUT_X.to_string()],
            output: vec![OUTPUT_Z.to_string()],
            name: "".to_string(),
            doc_string: "".to_string(),
        }],
        name: "".to_string(),
        initializer: vec![],
        input: vec![
            ValueInfoProto {
                name: INPUT_X.to_string(),
                doc_string: "".to_string(),
                r#type: None,
            },
            ValueInfoProto {
                name: INPUT_Y.to_string(),
                doc_string: "".to_string(),
                r#type: None,
            },
        ],
        output: vec![ValueInfoProto {
            name: OUTPUT_Z.to_string(),
            doc_string: "".to_string(),
            r#type: None,
        }],
        value_info: vec![],
        doc_string: "".to_string(),
        sparse_initializer: vec![],
        quantization_annotation: vec![],
    }));

    let x = Tensor::from_vec(
        //
        vec![1.0f32, 2.0f32, 3.0f32, 4.0f32],
        &[2, 2],
        &Device::Cpu,
    )?;

    let mut inputs: HashMap<String, Tensor> = HashMap::new();
    inputs.insert(INPUT_X.to_string(), x);

    let eval = candle_onnx::simple_eval(&manual_graph, inputs)?;
    assert_eq!(eval.len(), 1);

    let z = eval.get(OUTPUT_Z).expect("Output 'z' not found");

    let results = z.to_vec2::<f32>()?;

    assert_eq!(
        results,
        vec![vec![0.26894143, 0.7310586], vec![0.26894143, 0.7310586]]
    );

    Ok(())
}

// "Softmax"
#[test]
fn test_softmax_operation() -> Result<()> {
    let manual_graph = create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: "Softmax".to_string(),
            domain: "".to_string(),
            attribute: vec![],
            input: vec![INPUT_X.to_string()],
            output: vec![OUTPUT_Z.to_string()],
            name: "".to_string(),
            doc_string: "".to_string(),
        }],
        name: "".to_string(),
        initializer: vec![],
        input: vec![
            ValueInfoProto {
                name: INPUT_X.to_string(),
                doc_string: "".to_string(),
                r#type: None,
            },
            ValueInfoProto {
                name: INPUT_Y.to_string(),
                doc_string: "".to_string(),
                r#type: None,
            },
        ],
        output: vec![ValueInfoProto {
            name: OUTPUT_Z.to_string(),
            doc_string: "".to_string(),
            r#type: None,
        }],
        value_info: vec![],
        doc_string: "".to_string(),
        sparse_initializer: vec![],
        quantization_annotation: vec![],
    }));

    let x = Tensor::from_vec(
        //
        vec![1.0f32, 2.0f32, 3.0f32, 4.0f32],
        &[2, 2],
        &Device::Cpu,
    )?;

    let mut inputs: HashMap<String, Tensor> = HashMap::new();
    inputs.insert(INPUT_X.to_string(), x);

    let eval = candle_onnx::simple_eval(&manual_graph, inputs)?;
    assert_eq!(eval.len(), 1);

    let z = eval.get(OUTPUT_Z).expect("Output 'z' not found");

    let results = z.to_vec2::<f32>()?;

    assert_eq!(
        results,
        vec![vec![0.26894143, 0.7310586], vec![0.26894143, 0.7310586]]
    );

    Ok(())
}

// "Transpose"
#[test]
fn test_transpose_operation() -> Result<()> {
    let manual_graph = create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: "Transpose".to_string(),
            domain: "".to_string(),
            attribute: vec![],
            input: vec![INPUT_X.to_string()],
            output: vec![OUTPUT_Z.to_string()],
            name: "".to_string(),
            doc_string: "".to_string(),
        }],
        name: "".to_string(),
        initializer: vec![],
        input: vec![
            ValueInfoProto {
                name: INPUT_X.to_string(),
                doc_string: "".to_string(),
                r#type: None,
            },
            ValueInfoProto {
                name: INPUT_Y.to_string(),
                doc_string: "".to_string(),
                r#type: None,
            },
        ],
        output: vec![ValueInfoProto {
            name: OUTPUT_Z.to_string(),
            doc_string: "".to_string(),
            r#type: None,
        }],
        value_info: vec![],
        doc_string: "".to_string(),
        sparse_initializer: vec![],
        quantization_annotation: vec![],
    }));

    let x = Tensor::from_vec(
        //
        vec![1.0f32, 2.0f32, 3.0f32, 4.0f32],
        &[2, 2],
        &Device::Cpu,
    )?;

    let mut inputs: HashMap<String, Tensor> = HashMap::new();
    inputs.insert(INPUT_X.to_string(), x);

    let eval = candle_onnx::simple_eval(&manual_graph, inputs)?;
    assert_eq!(eval.len(), 1);

    let z = eval.get(OUTPUT_Z).expect("Output 'z' not found");

    let results = z.to_vec2::<f32>()?;

    assert_eq!(results, vec![vec![1.0, 3.0], vec![2.0, 4.0]]);

    Ok(())
}

// "Dropout"
#[test]
fn test_dropout_operation() -> Result<()> {
    let manual_graph = create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: "Dropout".to_string(),
            domain: "".to_string(),
            attribute: vec![],
            input: vec![INPUT_X.to_string()],
            output: vec![OUTPUT_Z.to_string()],
            name: "".to_string(),
            doc_string: "".to_string(),
        }],
        name: "".to_string(),
        initializer: vec![],
        input: vec![
            ValueInfoProto {
                name: INPUT_X.to_string(),
                doc_string: "".to_string(),
                r#type: None,
            },
            ValueInfoProto {
                name: INPUT_Y.to_string(),
                doc_string: "".to_string(),
                r#type: None,
            },
        ],
        output: vec![ValueInfoProto {
            name: OUTPUT_Z.to_string(),
            doc_string: "".to_string(),
            r#type: None,
        }],
        value_info: vec![],
        doc_string: "".to_string(),
        sparse_initializer: vec![],
        quantization_annotation: vec![],
    }));
    let x = Tensor::from_vec(
        //
        vec![1.0f32, 2.0f32, 3.0f32, 4.0f32],
        &[2, 2],
        &Device::Cpu,
    )?;

    let mut inputs: HashMap<String, Tensor> = HashMap::new();
    inputs.insert(INPUT_X.to_string(), x);

    let eval = candle_onnx::simple_eval(&manual_graph, inputs)?;
    assert_eq!(eval.len(), 1);

    let z = eval.get(OUTPUT_Z).expect("Output 'z' not found");

    let results = z.to_vec2::<f32>()?;

    assert_eq!(results, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);

    Ok(())
}

// "Flatten"
#[test]
fn test_flatten_operation() -> Result<()> {
    let mut att_axis = AttributeProto {
        name: "axis".to_string(),
        ref_attr_name: "axis".to_string(),
        i: 0,
        doc_string: "axis".to_string(),
        r#type: 2,
        f: 0.0,
        s: vec![],
        t: None,
        g: None,
        sparse_tensor: None,
        tp: None,
        floats: vec![],
        ints: vec![],
        strings: vec![],
        tensors: vec![],
        graphs: vec![],
        sparse_tensors: vec![],
        type_protos: vec![],
    };
    let manual_graph = create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: "Flatten".to_string(),
            domain: "".to_string(),
            attribute: vec![att_axis.clone()],
            input: vec![INPUT_X.to_string()],
            output: vec![OUTPUT_Z.to_string()],
            name: "".to_string(),
            doc_string: "".to_string(),
        }],
        name: "".to_string(),
        initializer: vec![],
        input: vec![
            ValueInfoProto {
                name: INPUT_X.to_string(),
                doc_string: "".to_string(),
                r#type: None,
            },
            ValueInfoProto {
                name: INPUT_Y.to_string(),
                doc_string: "".to_string(),
                r#type: None,
            },
        ],
        output: vec![ValueInfoProto {
            name: OUTPUT_Z.to_string(),
            doc_string: "".to_string(),
            r#type: None,
        }],
        value_info: vec![],
        doc_string: "".to_string(),
        sparse_initializer: vec![],
        quantization_annotation: vec![],
    }));
    let x = Tensor::from_vec(
        vec![
            1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32, 7.0f32, 8.0f32,
        ],
        &[2, 2, 2],
        &Device::Cpu,
    )?;

    let mut inputs: HashMap<String, Tensor> = HashMap::new();
    inputs.insert(INPUT_X.to_string(), x);

    let eval = candle_onnx::simple_eval(&manual_graph, inputs.clone())?;
    assert_eq!(eval.len(), 1);

    let z = eval.get(OUTPUT_Z).expect("Output 'z' not found");

    let results = z.to_vec2::<f32>()?;

    assert_eq!(results, vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]);

    att_axis.i = 1;
    let manual_graph = create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: "Flatten".to_string(),
            domain: "".to_string(),
            attribute: vec![att_axis.clone()],
            input: vec![INPUT_X.to_string()],
            output: vec![OUTPUT_Z.to_string()],
            name: "".to_string(),
            doc_string: "".to_string(),
        }],
        name: "".to_string(),
        initializer: vec![],
        input: vec![
            ValueInfoProto {
                name: INPUT_X.to_string(),
                doc_string: "".to_string(),
                r#type: None,
            },
            ValueInfoProto {
                name: INPUT_Y.to_string(),
                doc_string: "".to_string(),
                r#type: None,
            },
        ],
        output: vec![ValueInfoProto {
            name: OUTPUT_Z.to_string(),
            doc_string: "".to_string(),
            r#type: None,
        }],
        value_info: vec![],
        doc_string: "".to_string(),
        sparse_initializer: vec![],
        quantization_annotation: vec![],
    }));

    let eval = candle_onnx::simple_eval(&manual_graph, inputs)?;
    assert_eq!(eval.len(), 1);

    let z = eval.get(OUTPUT_Z).expect("Output 'z' not found");

    let results = z.to_vec2::<f32>()?;

    assert_eq!(
        results,
        vec![vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]]
    );

    Ok(())
}

// Below are ops that are implemented but not tested yet

// "MaxPool"
// #[test]

// "AveragePool"
// #[test]

// "BatchNormalization"
// #[test]

// "Squeeze"
// #[test]

// "ConstantOfShape"
#[test]
fn test_constant_of_shape() -> Result<()> {
    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-31
    test(&[4i64, 3, 2], Some(1.), &[1., 1., 1.])?;

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-31
    test(&[0.], Some(0i64), &[0i64])?;

    // "value" defaults to 0 f32
    test(&[1i64, 2, 3, 4], None as Option<i64>, &[0., 0., 0., 0.])?;

    fn test(
        input: impl NdArray,
        value: Option<impl NdArray>,
        expected: impl NdArray,
    ) -> Result<()> {
        let mut attribute = vec![];

        if let Some(value) = value {
            let tensor = Tensor::new(value, &Device::Cpu)?;

            let (value, data_type) = match tensor.dtype() {
                DType::U8 => (
                    tensor.to_vec0::<u8>()?.to_le_bytes().to_vec(),
                    DataType::Uint8,
                ),
                DType::U32 => (
                    tensor.to_vec0::<u32>()?.to_le_bytes().to_vec(),
                    DataType::Uint32,
                ),
                DType::I64 => (
                    tensor.to_vec0::<i64>()?.to_le_bytes().to_vec(),
                    DataType::Int64,
                ),
                DType::F32 => (
                    tensor.to_vec0::<f32>()?.to_le_bytes().to_vec(),
                    DataType::Float,
                ),
                DType::F64 => (
                    tensor.to_vec0::<f64>()?.to_le_bytes().to_vec(),
                    DataType::Double,
                ),
                _ => panic!("unsupported DType in test"),
            };
            let tensor = TensorProto {
                data_type: data_type.into(),
                dims: tensor.dims().iter().map(|v| *v as i64).collect(),
                raw_data: value,
                segment: None,
                float_data: vec![],
                int32_data: vec![],
                string_data: vec![],
                int64_data: vec![],
                name: "".to_string(),
                doc_string: "".to_string(),
                external_data: vec![],
                data_location: 0,
                double_data: vec![],
                uint64_data: vec![],
            };

            attribute.push(AttributeProto {
                name: "value".to_string(),
                ref_attr_name: "value".to_string(),
                i: 0,
                doc_string: "value".to_string(),
                r#type: AttributeType::Tensor.into(),
                f: 0.0,
                s: vec![],
                t: Some(tensor),
                g: None,
                sparse_tensor: None,
                tp: None,
                floats: vec![],
                ints: vec![],
                strings: vec![],
                tensors: vec![],
                graphs: vec![],
                sparse_tensors: vec![],
                type_protos: vec![],
            })
        }

        let manual_graph = create_model_proto_with_graph(Some(GraphProto {
            node: vec![NodeProto {
                op_type: "ConstantOfShape".to_string(),
                domain: "".to_string(),
                attribute,
                input: vec![INPUT_X.to_string()],
                output: vec![OUTPUT_Z.to_string()],
                name: "".to_string(),
                doc_string: "".to_string(),
            }],
            name: "".to_string(),
            initializer: vec![],
            input: vec![],
            output: vec![ValueInfoProto {
                name: OUTPUT_Z.to_string(),
                doc_string: "".to_string(),
                r#type: None,
            }],
            value_info: vec![],
            doc_string: "".to_string(),
            sparse_initializer: vec![],
            quantization_annotation: vec![],
        }));

        let mut inputs: HashMap<String, Tensor> = HashMap::new();
        inputs.insert(INPUT_X.to_string(), Tensor::new(input, &Device::Cpu)?);

        let eval = candle_onnx::simple_eval(&manual_graph, inputs)?;
        assert_eq!(eval.len(), 1);

        let z = eval
            .get(OUTPUT_Z)
            .expect("Output 'z' not found")
            .to_dtype(DType::F64)?;

        let expected = Tensor::new(expected, &Device::Cpu)?.to_dtype(DType::F64)?;
        match expected.dims().len() {
            0 => assert_eq!(z.to_vec0::<f64>()?, expected.to_vec0::<f64>()?),
            1 => assert_eq!(z.to_vec1::<f64>()?, expected.to_vec1::<f64>()?),
            2 => assert_eq!(z.to_vec2::<f64>()?, expected.to_vec2::<f64>()?),
            3 => assert_eq!(z.to_vec3::<f64>()?, expected.to_vec3::<f64>()?),
            _ => unreachable!(),
        };

        Ok(())
    }
    Ok(())
}

// "Unsqueeze"
// #[test]

// "Clip"
// #[test]

// "Gather"
#[test]
fn test_gather_operation() -> Result<()> {
    // test taken from https://onnx.ai/onnx/operators/onnx__Gather.html#summary.
    test(
        &[[1.0, 1.2], [2.3, 3.4], [4.5, 5.7]],
        &[[0i64, 1], [1, 2]],
        0,
        &[[[1.0, 1.2], [2.3, 3.4]], [[2.3, 3.4], [4.5, 5.7]]],
    )?;

    // test taken from https://onnx.ai/onnx/operators/onnx__Gather.html#summary.
    test(
        &[[1.0, 1.2, 1.9], [2.3, 3.4, 3.9], [4.5, 5.7, 5.9]],
        &[[0i64, 2]],
        1,
        &[[[1.0, 1.9]], [[2.3, 3.9]], [[4.5, 5.9]]],
    )?;

    // all the tests below are generated from numpy.take, which works like
    // onnx's Gather operation.
    test(&[1.0, 2.0, 3.0, 4.0], 3i64, 0, 4.0)?;

    test(&[[1.0, 2.0, 3.0, 4.0]], 3i64, 1, &[4.0])?;

    test(
        &[[1.0], [2.0], [3.0], [4.0]],
        &[3i64, 2],
        0,
        &[[4.0], [3.0]],
    )?;

    test(
        &[
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
            [[9.0, 10.0], [11.0, 12.0]],
            [[13.0, 14.0], [15.0, 16.0]],
        ],
        1i64,
        0,
        &[[5.0, 6.0], [7.0, 8.0]],
    )?;

    test(
        &[
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
            [[9.0, 10.0], [11.0, 12.0]],
            [[13.0, 14.0], [15.0, 16.0]],
        ],
        &[1i64, 0],
        0,
        &[[[5.0, 6.0], [7.0, 8.0]], [[1.0, 2.0], [3.0, 4.0]]],
    )?;

    fn test(
        data: impl NdArray,
        indices: impl NdArray,
        axis: i64,
        expected: impl NdArray,
    ) -> Result<()> {
        let att_axis = AttributeProto {
            name: "axis".to_string(),
            ref_attr_name: "axis".to_string(),
            i: axis,
            doc_string: "axis".to_string(),
            r#type: 2,
            f: 0.0,
            s: vec![],
            t: None,
            g: None,
            sparse_tensor: None,
            tp: None,
            floats: vec![],
            ints: vec![],
            strings: vec![],
            tensors: vec![],
            graphs: vec![],
            sparse_tensors: vec![],
            type_protos: vec![],
        };

        let manual_graph = create_model_proto_with_graph(Some(GraphProto {
            node: vec![NodeProto {
                op_type: "Gather".to_string(),
                domain: "".to_string(),
                attribute: vec![att_axis],
                input: vec![INPUT_X.to_string(), INPUT_Y.to_string()],
                output: vec![OUTPUT_Z.to_string()],
                name: "".to_string(),
                doc_string: "".to_string(),
            }],
            name: "".to_string(),
            initializer: vec![],
            input: vec![],
            output: vec![ValueInfoProto {
                name: OUTPUT_Z.to_string(),
                doc_string: "".to_string(),
                r#type: None,
            }],
            value_info: vec![],
            doc_string: "".to_string(),
            sparse_initializer: vec![],
            quantization_annotation: vec![],
        }));

        let mut inputs: HashMap<String, Tensor> = HashMap::new();
        inputs.insert(INPUT_X.to_string(), Tensor::new(data, &Device::Cpu)?);
        inputs.insert(INPUT_Y.to_string(), Tensor::new(indices, &Device::Cpu)?);

        let eval = candle_onnx::simple_eval(&manual_graph, inputs)?;
        assert_eq!(eval.len(), 1);

        let z = eval.get(OUTPUT_Z).expect("Output 'z' not found");

        let expected = Tensor::new(expected, &Device::Cpu)?;
        match expected.dims().len() {
            0 => assert_eq!(z.to_vec0::<f64>()?, expected.to_vec0::<f64>()?),
            1 => assert_eq!(z.to_vec1::<f64>()?, expected.to_vec1::<f64>()?),
            2 => assert_eq!(z.to_vec2::<f64>()?, expected.to_vec2::<f64>()?),
            3 => assert_eq!(z.to_vec3::<f64>()?, expected.to_vec3::<f64>()?),
            _ => unreachable!(),
        };

        Ok(())
    }
    Ok(())
}

// "Shape"
#[test]
fn test_shape_operation() -> Result<()> {
    let manual_graph = create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: "Shape".to_string(),
            domain: "".to_string(),
            attribute: vec![],
            input: vec![INPUT_X.to_string()],
            output: vec![OUTPUT_Z.to_string()],
            name: "".to_string(),
            doc_string: "".to_string(),
        }],
        name: "".to_string(),
        initializer: vec![],
        input: vec![],
        output: vec![ValueInfoProto {
            name: OUTPUT_Z.to_string(),
            doc_string: "".to_string(),
            r#type: None,
        }],
        value_info: vec![ValueInfoProto {
            name: INPUT_X.to_string(),
            doc_string: "".to_string(),
            r#type: None,
        }],
        doc_string: "".to_string(),
        sparse_initializer: vec![],
        quantization_annotation: vec![],
    }));
    let x = Tensor::from_vec(vec![1.0f32, 2.0f32, 3.0f32, 4.0f32], &[2, 2], &Device::Cpu)?;

    let mut inputs: HashMap<String, Tensor> = HashMap::new();
    inputs.insert(INPUT_X.to_string(), x);

    let eval = candle_onnx::simple_eval(&manual_graph, inputs)?;
    assert_eq!(eval.len(), 1);

    let z = eval.get(OUTPUT_Z).expect("Output 'z' not found");
    let results = z.to_vec1::<i64>()?;
    assert_eq!(results, vec![2, 2]);

    Ok(())
}

// "Conv"
// #[test]

// "Concat"
// #[test]

// "Abs"
#[test]
fn test_abs_operation() -> Result<()> {
    let manual_graph = create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: "Abs".to_string(),
            domain: "".to_string(),
            attribute: vec![],
            input: vec![INPUT_X.to_string()],
            output: vec![OUTPUT_Z.to_string()],
            name: "".to_string(),
            doc_string: "".to_string(),
        }],
        name: "".to_string(),
        initializer: vec![],
        input: vec![
            ValueInfoProto {
                name: INPUT_X.to_string(),
                doc_string: "".to_string(),
                r#type: None,
            },
            ValueInfoProto {
                name: INPUT_Y.to_string(),
                doc_string: "".to_string(),
                r#type: None,
            },
        ],
        output: vec![ValueInfoProto {
            name: OUTPUT_Z.to_string(),
            doc_string: "".to_string(),
            r#type: None,
        }],
        value_info: vec![],
        doc_string: "".to_string(),
        sparse_initializer: vec![],
        quantization_annotation: vec![],
    }));
    let x = Tensor::from_vec(
        vec![-1.0f32, 2.0f32, -3.0f32, 4.0f32],
        &[2, 2],
        &Device::Cpu,
    )?;

    let mut inputs: HashMap<String, Tensor> = HashMap::new();
    inputs.insert(INPUT_X.to_string(), x);

    let eval = candle_onnx::simple_eval(&manual_graph, inputs)?;
    assert_eq!(eval.len(), 1);

    let z = eval.get(OUTPUT_Z).expect("Output 'z' not found");

    let results = z.to_vec2::<f32>()?;

    assert_eq!(results, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);

    Ok(())
}

// "Cos"
#[test]
fn test_cos_operation() -> Result<()> {
    let manual_graph = create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: "Cos".to_string(),
            domain: "".to_string(),
            attribute: vec![],
            input: vec![INPUT_X.to_string()],
            output: vec![OUTPUT_Z.to_string()],
            name: "".to_string(),
            doc_string: "".to_string(),
        }],
        name: "".to_string(),
        initializer: vec![],
        input: vec![
            ValueInfoProto {
                name: INPUT_X.to_string(),
                doc_string: "".to_string(),
                r#type: None,
            },
            ValueInfoProto {
                name: INPUT_Y.to_string(),
                doc_string: "".to_string(),
                r#type: None,
            },
        ],
        output: vec![ValueInfoProto {
            name: OUTPUT_Z.to_string(),
            doc_string: "".to_string(),
            r#type: None,
        }],
        value_info: vec![],
        doc_string: "".to_string(),
        sparse_initializer: vec![],
        quantization_annotation: vec![],
    }));
    let x = Tensor::from_vec(vec![0.0f32, 1.0f32, 2.0f32, 3.0f32], &[2, 2], &Device::Cpu)?;

    let mut inputs: HashMap<String, Tensor> = HashMap::new();
    inputs.insert(INPUT_X.to_string(), x);

    let eval = candle_onnx::simple_eval(&manual_graph, inputs)?;
    assert_eq!(eval.len(), 1);

    let z = eval.get(OUTPUT_Z).expect("Output 'z' not found");
    assert_eq!(to_vec2_round(z, 4)?, [[1.0, 0.5403], [-0.4161, -0.99]]);
    Ok(())
}

// "Sin"
#[test]
fn test_sin_operation() -> Result<()> {
    let manual_graph = create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: "Sin".to_string(),
            domain: "".to_string(),
            attribute: vec![],
            input: vec![INPUT_X.to_string()],
            output: vec![OUTPUT_Z.to_string()],
            name: "".to_string(),
            doc_string: "".to_string(),
        }],
        name: "".to_string(),
        initializer: vec![],
        input: vec![
            ValueInfoProto {
                name: INPUT_X.to_string(),
                doc_string: "".to_string(),
                r#type: None,
            },
            ValueInfoProto {
                name: INPUT_Y.to_string(),
                doc_string: "".to_string(),
                r#type: None,
            },
        ],
        output: vec![ValueInfoProto {
            name: OUTPUT_Z.to_string(),
            doc_string: "".to_string(),
            r#type: None,
        }],
        value_info: vec![],
        doc_string: "".to_string(),
        sparse_initializer: vec![],
        quantization_annotation: vec![],
    }));
    let x = Tensor::from_vec(vec![0.0f32, 1.0f32, 2.0f32, 3.0f32], &[2, 2], &Device::Cpu)?;
    let mut inputs: HashMap<String, Tensor> = HashMap::new();
    inputs.insert(INPUT_X.to_string(), x);
    let eval = candle_onnx::simple_eval(&manual_graph, inputs)?;
    assert_eq!(eval.len(), 1);
    let z = eval.get(OUTPUT_Z).expect("Output 'z' not found");
    assert_eq!(to_vec2_round(z, 4)?, [[0.0, 0.8415], [0.9093, 0.1411]]);
    Ok(())
}

// "Neg"
#[test]
fn test_neg_operation() -> Result<()> {
    let manual_graph = create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: "Neg".to_string(),
            domain: "".to_string(),
            attribute: vec![],
            input: vec![INPUT_X.to_string()],
            output: vec![OUTPUT_Z.to_string()],
            name: "".to_string(),
            doc_string: "".to_string(),
        }],
        name: "".to_string(),
        initializer: vec![],
        input: vec![
            ValueInfoProto {
                name: INPUT_X.to_string(),
                doc_string: "".to_string(),
                r#type: None,
            },
            ValueInfoProto {
                name: INPUT_Y.to_string(),
                doc_string: "".to_string(),
                r#type: None,
            },
        ],
        output: vec![ValueInfoProto {
            name: OUTPUT_Z.to_string(),
            doc_string: "".to_string(),
            r#type: None,
        }],
        value_info: vec![],
        doc_string: "".to_string(),
        sparse_initializer: vec![],
        quantization_annotation: vec![],
    }));
    let x = Tensor::from_vec(vec![1.0f32, 2.0f32, 3.0f32, 4.0f32], &[2, 2], &Device::Cpu)?;

    let mut inputs: HashMap<String, Tensor> = HashMap::new();
    inputs.insert(INPUT_X.to_string(), x);

    let eval = candle_onnx::simple_eval(&manual_graph, inputs)?;
    assert_eq!(eval.len(), 1);

    let z = eval.get(OUTPUT_Z).expect("Output 'z' not found");

    let results = z.to_vec2::<f32>()?;

    assert_eq!(results, vec![vec![-1.0, -2.0], vec![-3.0, -4.0]]);

    Ok(())
}

// "Erf"
// #[test]

// "Tanh"
#[test]
fn test_tanh_operation() -> Result<()> {
    let manual_graph = create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: "Tanh".to_string(),
            domain: "".to_string(),
            attribute: vec![],
            input: vec![INPUT_X.to_string()],
            output: vec![OUTPUT_Z.to_string()],
            name: "".to_string(),
            doc_string: "".to_string(),
        }],
        name: "".to_string(),
        initializer: vec![],
        input: vec![
            ValueInfoProto {
                name: INPUT_X.to_string(),
                doc_string: "".to_string(),
                r#type: None,
            },
            ValueInfoProto {
                name: INPUT_Y.to_string(),
                doc_string: "".to_string(),
                r#type: None,
            },
        ],
        output: vec![ValueInfoProto {
            name: OUTPUT_Z.to_string(),
            doc_string: "".to_string(),
            r#type: None,
        }],
        value_info: vec![],
        doc_string: "".to_string(),
        sparse_initializer: vec![],
        quantization_annotation: vec![],
    }));
    let x = Tensor::from_vec(vec![0.0f32, 1.0f32, 2.0f32, 3.0f32], &[2, 2], &Device::Cpu)?;

    let mut inputs: HashMap<String, Tensor> = HashMap::new();
    inputs.insert(INPUT_X.to_string(), x);

    let eval = candle_onnx::simple_eval(&manual_graph, inputs)?;
    assert_eq!(eval.len(), 1);

    let z = eval.get(OUTPUT_Z).expect("Output 'z' not found");

    let results = z.to_vec2::<f32>()?;

    assert_eq!(
        results,
        vec![vec![0.0, 0.7615942], vec![0.9640276, 0.9950548]]
    );

    Ok(())
}

// "Sigmoid"
#[test]
fn test_sigmoid_operation() -> Result<()> {
    let manual_graph = create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: "Sigmoid".to_string(),
            domain: "".to_string(),
            attribute: vec![],
            input: vec![INPUT_X.to_string()],
            output: vec![OUTPUT_Z.to_string()],
            name: "".to_string(),
            doc_string: "".to_string(),
        }],
        name: "".to_string(),
        initializer: vec![],
        input: vec![
            ValueInfoProto {
                name: INPUT_X.to_string(),
                doc_string: "".to_string(),
                r#type: None,
            },
            ValueInfoProto {
                name: INPUT_Y.to_string(),
                doc_string: "".to_string(),
                r#type: None,
            },
        ],
        output: vec![ValueInfoProto {
            name: OUTPUT_Z.to_string(),
            doc_string: "".to_string(),
            r#type: None,
        }],
        value_info: vec![],
        doc_string: "".to_string(),
        sparse_initializer: vec![],
        quantization_annotation: vec![],
    }));
    let x = Tensor::from_vec(vec![0.0f32, 1.0f32, 2.0f32, 3.0f32], &[2, 2], &Device::Cpu)?;

    let mut inputs: HashMap<String, Tensor> = HashMap::new();
    inputs.insert(INPUT_X.to_string(), x);

    let eval = candle_onnx::simple_eval(&manual_graph, inputs)?;
    assert_eq!(eval.len(), 1);

    let z = eval.get(OUTPUT_Z).expect("Output 'z' not found");

    let results = z.to_vec2::<f32>()?;

    assert_eq!(
        results,
        vec![vec![0.5, 0.7310586], vec![0.880797, 0.95257413]]
    );

    Ok(())
}

// "Gelu"
#[test]
fn test_gelu_operation() -> Result<()> {
    let manual_graph = create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: "Gelu".to_string(),
            domain: "".to_string(),
            attribute: vec![],
            input: vec![INPUT_X.to_string()],
            output: vec![OUTPUT_Z.to_string()],
            name: "".to_string(),
            doc_string: "".to_string(),
        }],
        name: "".to_string(),
        initializer: vec![],
        input: vec![
            ValueInfoProto {
                name: INPUT_X.to_string(),
                doc_string: "".to_string(),
                r#type: None,
            },
            ValueInfoProto {
                name: INPUT_Y.to_string(),
                doc_string: "".to_string(),
                r#type: None,
            },
        ],
        output: vec![ValueInfoProto {
            name: OUTPUT_Z.to_string(),
            doc_string: "".to_string(),
            r#type: None,
        }],
        value_info: vec![],
        doc_string: "".to_string(),
        sparse_initializer: vec![],
        quantization_annotation: vec![],
    }));
    let x = Tensor::from_vec(vec![0.0f32, 1.0f32, 2.0f32, 3.0f32], &[2, 2], &Device::Cpu)?;

    let mut inputs: HashMap<String, Tensor> = HashMap::new();
    inputs.insert(INPUT_X.to_string(), x);

    let eval = candle_onnx::simple_eval(&manual_graph, inputs)?;
    assert_eq!(eval.len(), 1);

    let z = eval.get(OUTPUT_Z).expect("Output 'z' not found");

    let results = z.to_vec2::<f32>()?;

    assert_eq!(
        results,
        vec![vec![0.0, 0.8413448], vec![1.9544997, 2.9959502]]
    );

    Ok(())
}

// "Relu"
#[test]
fn test_relu_operation() -> Result<()> {
    let manual_graph = create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: "Relu".to_string(),
            domain: "".to_string(),
            attribute: vec![],
            input: vec![INPUT_X.to_string()],
            output: vec![OUTPUT_Z.to_string()],
            name: "".to_string(),
            doc_string: "".to_string(),
        }],
        name: "".to_string(),
        initializer: vec![],
        input: vec![ValueInfoProto {
            name: INPUT_X.to_string(),
            doc_string: "".to_string(),
            r#type: None,
        }],
        output: vec![ValueInfoProto {
            name: OUTPUT_Z.to_string(),
            doc_string: "".to_string(),
            r#type: None,
        }],
        value_info: vec![],
        doc_string: "".to_string(),
        sparse_initializer: vec![],
        quantization_annotation: vec![],
    }));
    let x = Tensor::from_vec(
        vec![-1.0f32, 1.0f32, -2.0f32, 3.0f32],
        &[2, 2],
        &Device::Cpu,
    )?;

    let mut inputs: HashMap<String, Tensor> = HashMap::new();
    inputs.insert(INPUT_X.to_string(), x);

    let eval = candle_onnx::simple_eval(&manual_graph, inputs)?;
    assert_eq!(eval.len(), 1);

    let z = eval.get(OUTPUT_Z).expect("Output 'z' not found");

    let results = z.to_vec2::<f32>()?;

    assert_eq!(results, vec![vec![0.0, 1.0], vec![0.0, 3.0]]);

    Ok(())
}

// "Constant"
// #[test]

// "Cast"
// #[test]

// "ReduceMean"
#[test]
fn test_reduce_mean() -> Result<()> {
    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-120 default_axes_keepdims
    test(
        &[
            [[5., 1.], [20., 2.]],
            [[30., 1.], [40., 2.]],
            [[55., 1.], [60., 2.]],
        ],
        None,
        1,
        &[[[18.25]]],
    )?;

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-120 do_no_keepdims
    test(
        &[
            [[5., 1.], [20., 2.]],
            [[30., 1.], [40., 2.]],
            [[55., 1.], [60., 2.]],
        ],
        Some(vec![1]),
        0,
        &[[12.5, 1.5], [35.0, 1.5], [57.5, 1.5]],
    )?;

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-120 keepdims
    test(
        &[
            [[5., 1.], [20., 2.]],
            [[30., 1.], [40., 2.]],
            [[55., 1.], [60., 2.]],
        ],
        Some(vec![1]),
        1,
        &[[[12.5, 1.5]], [[35.0, 1.5]], [[57.5, 1.5]]],
    )?;

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-120 negative_axes_keepdims
    test(
        &[
            [[5., 1.], [20., 2.]],
            [[30., 1.], [40., 2.]],
            [[55., 1.], [60., 2.]],
        ],
        Some(vec![-2]),
        1,
        &[[[12.5, 1.5]], [[35.0, 1.5]], [[57.5, 1.5]]],
    )?;

    // All the test data below was generated based on numpy's np.mean
    test(
        &[
            [[5., 1.], [20., 2.]],
            [[30., 1.], [40., 2.]],
            [[55., 1.], [60., 2.]],
        ],
        Some(vec![1, 2]),
        0,
        &[7.0, 18.25, 29.5],
    )?;

    test(
        &[
            [[5., 1.], [20., 2.]],
            [[30., 1.], [40., 2.]],
            [[55., 1.], [60., 2.]],
        ],
        Some(vec![1, 2]),
        1,
        &[[[7.0]], [[18.25]], [[29.5]]],
    )?;

    test(&[1., 2., 3.], None, 1, &[2.0])?;

    fn test(
        data: impl NdArray,
        axes: Option<Vec<i64>>,
        keepdims: i64,
        expected: impl NdArray,
    ) -> Result<()> {
        let has_axes = axes.is_some();

        let att_axes = AttributeProto {
            name: "axes".to_string(),
            ref_attr_name: "axes".to_string(),
            i: 0,
            doc_string: "axes".to_string(),
            r#type: 7,
            f: 0.0,
            s: vec![],
            t: None,
            g: None,
            sparse_tensor: None,
            tp: None,
            floats: vec![],
            ints: axes.unwrap_or_default(),
            strings: vec![],
            tensors: vec![],
            graphs: vec![],
            sparse_tensors: vec![],
            type_protos: vec![],
        };

        let att_keepdims = AttributeProto {
            name: "keepdims".to_string(),
            ref_attr_name: "keepdims".to_string(),
            i: keepdims,
            doc_string: "keepdims".to_string(),
            r#type: 2,
            f: 0.0,
            s: vec![],
            t: None,
            g: None,
            sparse_tensor: None,
            tp: None,
            floats: vec![],
            ints: vec![],
            strings: vec![],
            tensors: vec![],
            graphs: vec![],
            sparse_tensors: vec![],
            type_protos: vec![],
        };

        let manual_graph = create_model_proto_with_graph(Some(GraphProto {
            node: vec![NodeProto {
                op_type: "ReduceMean".to_string(),
                domain: "".to_string(),
                attribute: if has_axes {
                    vec![att_axes, att_keepdims]
                } else {
                    vec![att_keepdims]
                },
                input: vec![INPUT_X.to_string()],
                output: vec![OUTPUT_Z.to_string()],
                name: "".to_string(),
                doc_string: "".to_string(),
            }],
            name: "".to_string(),
            initializer: vec![],
            input: vec![],
            output: vec![ValueInfoProto {
                name: OUTPUT_Z.to_string(),
                doc_string: "".to_string(),
                r#type: None,
            }],
            value_info: vec![],
            doc_string: "".to_string(),
            sparse_initializer: vec![],
            quantization_annotation: vec![],
        }));

        let mut inputs: HashMap<String, Tensor> = HashMap::new();
        inputs.insert(INPUT_X.to_string(), Tensor::new(data, &Device::Cpu)?);

        let eval = candle_onnx::simple_eval(&manual_graph, inputs)?;
        assert_eq!(eval.len(), 1);

        let z = eval.get(OUTPUT_Z).expect("Output 'z' not found");

        let expected = Tensor::new(expected, &Device::Cpu)?;
        match expected.dims().len() {
            0 => assert_eq!(z.to_vec0::<f64>()?, expected.to_vec0::<f64>()?),
            1 => assert_eq!(z.to_vec1::<f64>()?, expected.to_vec1::<f64>()?),
            2 => assert_eq!(z.to_vec2::<f64>()?, expected.to_vec2::<f64>()?),
            3 => assert_eq!(z.to_vec3::<f64>()?, expected.to_vec3::<f64>()?),
            _ => unreachable!(),
        };

        Ok(())
    }

    Ok(())
}

// "Sqrt"
#[test]
fn test_sqrt() -> Result<()> {
    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-155
    test(&[1., 4., 9.], &[1., 2., 3.])?;

    fn test(data: impl NdArray, expected: impl NdArray) -> Result<()> {
        let manual_graph = create_model_proto_with_graph(Some(GraphProto {
            node: vec![NodeProto {
                op_type: "Sqrt".to_string(),
                domain: "".to_string(),
                attribute: vec![],
                input: vec![INPUT_X.to_string()],
                output: vec![OUTPUT_Z.to_string()],
                name: "".to_string(),
                doc_string: "".to_string(),
            }],
            name: "".to_string(),
            initializer: vec![],
            input: vec![],
            output: vec![ValueInfoProto {
                name: OUTPUT_Z.to_string(),
                doc_string: "".to_string(),
                r#type: None,
            }],
            value_info: vec![],
            doc_string: "".to_string(),
            sparse_initializer: vec![],
            quantization_annotation: vec![],
        }));

        let mut inputs: HashMap<String, Tensor> = HashMap::new();
        inputs.insert(INPUT_X.to_string(), Tensor::new(data, &Device::Cpu)?);

        let eval = candle_onnx::simple_eval(&manual_graph, inputs)?;
        assert_eq!(eval.len(), 1);

        let z = eval.get(OUTPUT_Z).expect("Output 'z' not found");

        let expected = Tensor::new(expected, &Device::Cpu)?;
        match expected.dims().len() {
            0 => assert_eq!(z.to_vec0::<f64>()?, expected.to_vec0::<f64>()?),
            1 => assert_eq!(z.to_vec1::<f64>()?, expected.to_vec1::<f64>()?),
            2 => assert_eq!(z.to_vec2::<f64>()?, expected.to_vec2::<f64>()?),
            3 => assert_eq!(z.to_vec3::<f64>()?, expected.to_vec3::<f64>()?),
            _ => unreachable!(),
        };

        Ok(())
    }

    Ok(())
}

// "RandomUniform"
#[test]
fn test_random_uniform() -> Result<()> {
    test(vec![3, 2, 1, 4], None, None)?;
    test(vec![2, 2, 2, 2], Some(-10.0), None)?;
    test(vec![2, 2, 2, 2], None, Some(10.0))?;
    test(vec![1, 2, 3, 4], Some(-10.0), Some(10.0))?;

    fn test(shape: Vec<i64>, low: Option<f32>, high: Option<f32>) -> Result<()> {
        let att_low = AttributeProto {
            name: "low".to_string(),
            ref_attr_name: "low".to_string(),
            i: 0,
            doc_string: "low".to_string(),
            r#type: 1, // FLOAT
            f: low.unwrap_or(0.0),
            s: vec![],
            t: None,
            g: None,
            sparse_tensor: None,
            tp: None,
            floats: vec![],
            ints: vec![],
            strings: vec![],
            tensors: vec![],
            graphs: vec![],
            sparse_tensors: vec![],
            type_protos: vec![],
        };
        let att_high = AttributeProto {
            name: "high".to_string(),
            ref_attr_name: "high".to_string(),
            i: 0,
            doc_string: "high".to_string(),
            r#type: 1, // FLOAT
            f: high.unwrap_or(1.0),
            s: vec![],
            t: None,
            g: None,
            sparse_tensor: None,
            tp: None,
            floats: vec![],
            ints: vec![],
            strings: vec![],
            tensors: vec![],
            graphs: vec![],
            sparse_tensors: vec![],
            type_protos: vec![],
        };
        let att_shape = AttributeProto {
            name: "shape".to_string(),
            ref_attr_name: "shape".to_string(),
            i: 0,
            doc_string: "shape".to_string(),
            r#type: 7, // INTS
            f: 0.0,
            s: vec![],
            t: None,
            g: None,
            sparse_tensor: None,
            tp: None,
            floats: vec![],
            ints: shape,
            strings: vec![],
            tensors: vec![],
            graphs: vec![],
            sparse_tensors: vec![],
            type_protos: vec![],
        };
        let att_dtype = AttributeProto {
            name: "dtype".to_string(),
            ref_attr_name: "dtype".to_string(),
            i: 11, // DOUBLE
            doc_string: "dtype".to_string(),
            r#type: 2, // INT
            f: 0.0,
            s: vec![],
            t: None,
            g: None,
            sparse_tensor: None,
            tp: None,
            floats: vec![],
            ints: vec![],
            strings: vec![],
            tensors: vec![],
            graphs: vec![],
            sparse_tensors: vec![],
            type_protos: vec![],
        };
        let attrs = {
            let mut mut_attrs = vec![att_shape, att_dtype];
            if low.is_some() {
                mut_attrs.push(att_low);
            }
            if high.is_some() {
                mut_attrs.push(att_high);
            }
            mut_attrs
        };
        let manual_graph = create_model_proto_with_graph(Some(GraphProto {
            node: vec![NodeProto {
                op_type: "RandomUniform".to_string(),
                domain: "".to_string(),
                attribute: attrs,
                input: vec![],
                output: vec![OUTPUT_Z.to_string()],
                name: "".to_string(),
                doc_string: "".to_string(),
            }],
            name: "".to_string(),
            initializer: vec![],
            input: vec![],
            output: vec![ValueInfoProto {
                name: OUTPUT_Z.to_string(),
                doc_string: "".to_string(),
                r#type: None,
            }],
            value_info: vec![],
            doc_string: "".to_string(),
            sparse_initializer: vec![],
            quantization_annotation: vec![],
        }));
        let eval = candle_onnx::simple_eval(&manual_graph, HashMap::new())?;
        assert_eq!(eval.len(), 1);
        let z = eval.get(OUTPUT_Z).expect("Output 'z' not found");
        let min = z
            .flatten_all()?
            .to_vec1()?
            .into_iter()
            .reduce(f64::min)
            .unwrap();
        let max = z
            .flatten_all()?
            .to_vec1()?
            .into_iter()
            .reduce(f64::max)
            .unwrap();
        assert!(min >= low.unwrap_or(0.0).into());
        assert!(max <= high.unwrap_or(1.0).into());
        assert_ne!(min, max);
        Ok(())
    }

    Ok(())
}

// "RandomNormal"
#[test]
fn test_random_normal() -> Result<()> {
    test(vec![3, 2, 1, 4], None, None)?;
    test(vec![2, 2, 2, 2], Some(-10.0), None)?;
    test(vec![2, 2, 2, 2], None, Some(10.0))?;
    test(vec![1, 2, 3, 4], Some(-10.0), Some(10.0))?;

    fn test(shape: Vec<i64>, mean: Option<f32>, scale: Option<f32>) -> Result<()> {
        let att_mean = AttributeProto {
            name: "mean".to_string(),
            ref_attr_name: "mean".to_string(),
            i: 0,
            doc_string: "mean".to_string(),
            r#type: 1, // FLOAT
            f: mean.unwrap_or(0.0),
            s: vec![],
            t: None,
            g: None,
            sparse_tensor: None,
            tp: None,
            floats: vec![],
            ints: vec![],
            strings: vec![],
            tensors: vec![],
            graphs: vec![],
            sparse_tensors: vec![],
            type_protos: vec![],
        };
        let att_scale = AttributeProto {
            name: "scale".to_string(),
            ref_attr_name: "scale".to_string(),
            i: 0,
            doc_string: "scale".to_string(),
            r#type: 1, // FLOAT
            f: scale.unwrap_or(1.0),
            s: vec![],
            t: None,
            g: None,
            sparse_tensor: None,
            tp: None,
            floats: vec![],
            ints: vec![],
            strings: vec![],
            tensors: vec![],
            graphs: vec![],
            sparse_tensors: vec![],
            type_protos: vec![],
        };
        let att_shape = AttributeProto {
            name: "shape".to_string(),
            ref_attr_name: "shape".to_string(),
            i: 0,
            doc_string: "shape".to_string(),
            r#type: 7, // INTS
            f: 0.0,
            s: vec![],
            t: None,
            g: None,
            sparse_tensor: None,
            tp: None,
            floats: vec![],
            ints: shape,
            strings: vec![],
            tensors: vec![],
            graphs: vec![],
            sparse_tensors: vec![],
            type_protos: vec![],
        };
        let att_dtype = AttributeProto {
            name: "dtype".to_string(),
            ref_attr_name: "dtype".to_string(),
            i: 11, // DOUBLE
            doc_string: "dtype".to_string(),
            r#type: 2, // INT
            f: 0.0,
            s: vec![],
            t: None,
            g: None,
            sparse_tensor: None,
            tp: None,
            floats: vec![],
            ints: vec![],
            strings: vec![],
            tensors: vec![],
            graphs: vec![],
            sparse_tensors: vec![],
            type_protos: vec![],
        };
        let attrs = {
            let mut mut_attrs = vec![att_shape, att_dtype];
            if mean.is_some() {
                mut_attrs.push(att_mean);
            }
            if scale.is_some() {
                mut_attrs.push(att_scale);
            }
            mut_attrs
        };
        let manual_graph = create_model_proto_with_graph(Some(GraphProto {
            node: vec![NodeProto {
                op_type: "RandomNormal".to_string(),
                domain: "".to_string(),
                attribute: attrs,
                input: vec![],
                output: vec![OUTPUT_Z.to_string()],
                name: "".to_string(),
                doc_string: "".to_string(),
            }],
            name: "".to_string(),
            initializer: vec![],
            input: vec![],
            output: vec![ValueInfoProto {
                name: OUTPUT_Z.to_string(),
                doc_string: "".to_string(),
                r#type: None,
            }],
            value_info: vec![],
            doc_string: "".to_string(),
            sparse_initializer: vec![],
            quantization_annotation: vec![],
        }));
        let eval = candle_onnx::simple_eval(&manual_graph, HashMap::new())?;
        assert_eq!(eval.len(), 1);

        let z = eval.get(OUTPUT_Z).expect("Output 'z' not found");
        let data = z.flatten_all()?.to_vec1::<f64>()?;

        // test if values are unique
        for (i, a) in data.iter().enumerate() {
            for (j, b) in data.iter().enumerate() {
                if i == j {
                    continue;
                };
                assert_ne!(a, b);
            }
        }

        Ok(())
    }

    Ok(())
}

// "Range"
#[test]
fn test_range() -> Result<()> {
    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-113
    test(1., 5., 2., &[1., 3.])?;

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-113
    test(10i64, 6i64, -3i64, &[10i64, 7i64])?;

    fn test(
        start: impl NdArray,
        limit: impl NdArray,
        delta: impl NdArray,
        expected: impl NdArray,
    ) -> Result<()> {
        let manual_graph = create_model_proto_with_graph(Some(GraphProto {
            node: vec![NodeProto {
                op_type: "Range".to_string(),
                domain: "".to_string(),
                attribute: vec![],
                input: vec![
                    INPUT_X.to_string(),
                    INPUT_Y.to_string(),
                    INPUT_A.to_string(),
                ],
                output: vec![OUTPUT_Z.to_string()],
                name: "".to_string(),
                doc_string: "".to_string(),
            }],
            name: "".to_string(),
            initializer: vec![],
            input: vec![],
            output: vec![ValueInfoProto {
                name: OUTPUT_Z.to_string(),
                doc_string: "".to_string(),
                r#type: None,
            }],
            value_info: vec![],
            doc_string: "".to_string(),
            sparse_initializer: vec![],
            quantization_annotation: vec![],
        }));

        let mut inputs: HashMap<String, Tensor> = HashMap::new();
        inputs.insert(INPUT_X.to_string(), Tensor::new(start, &Device::Cpu)?);
        inputs.insert(INPUT_Y.to_string(), Tensor::new(limit, &Device::Cpu)?);
        inputs.insert(INPUT_A.to_string(), Tensor::new(delta, &Device::Cpu)?);

        let eval = candle_onnx::simple_eval(&manual_graph, inputs)?;
        assert_eq!(eval.len(), 1);

        let z = eval
            .get(OUTPUT_Z)
            .expect("Output 'z' not found")
            .to_dtype(DType::F64)?;

        let expected = Tensor::new(expected, &Device::Cpu)?.to_dtype(DType::F64)?;
        match expected.dims().len() {
            0 => assert_eq!(z.to_vec0::<f64>()?, expected.to_vec0::<f64>()?),
            1 => assert_eq!(z.to_vec1::<f64>()?, expected.to_vec1::<f64>()?),
            2 => assert_eq!(z.to_vec2::<f64>()?, expected.to_vec2::<f64>()?),
            3 => assert_eq!(z.to_vec3::<f64>()?, expected.to_vec3::<f64>()?),
            _ => unreachable!(),
        };

        Ok(())
    }

    Ok(())
}

// "Greater"
#[test]
fn test_greater() -> Result<()> {
    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-63
    test(&[1., 2., 3.], &[3., 2., 1.], &[0u8, 0, 1])?;

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-63
    test(&[1., 2., 3.], 2., &[0u8, 0, 1])?;

    fn test(a: impl NdArray, b: impl NdArray, expected: impl NdArray) -> Result<()> {
        let manual_graph = create_model_proto_with_graph(Some(GraphProto {
            node: vec![NodeProto {
                op_type: "Greater".to_string(),
                domain: "".to_string(),
                attribute: vec![],
                input: vec![INPUT_X.to_string(), INPUT_Y.to_string()],
                output: vec![OUTPUT_Z.to_string()],
                name: "".to_string(),
                doc_string: "".to_string(),
            }],
            name: "".to_string(),
            initializer: vec![],
            input: vec![],
            output: vec![ValueInfoProto {
                name: OUTPUT_Z.to_string(),
                doc_string: "".to_string(),
                r#type: None,
            }],
            value_info: vec![],
            doc_string: "".to_string(),
            sparse_initializer: vec![],
            quantization_annotation: vec![],
        }));

        let mut inputs: HashMap<String, Tensor> = HashMap::new();
        inputs.insert(INPUT_X.to_string(), Tensor::new(a, &Device::Cpu)?);
        inputs.insert(INPUT_Y.to_string(), Tensor::new(b, &Device::Cpu)?);

        let eval = candle_onnx::simple_eval(&manual_graph, inputs)?;
        assert_eq!(eval.len(), 1);

        let z = eval
            .get(OUTPUT_Z)
            .expect("Output 'z' not found")
            .to_dtype(DType::F64)?;

        let expected = Tensor::new(expected, &Device::Cpu)?.to_dtype(DType::F64)?;
        match expected.dims().len() {
            0 => assert_eq!(z.to_vec0::<f64>()?, expected.to_vec0::<f64>()?),
            1 => assert_eq!(z.to_vec1::<f64>()?, expected.to_vec1::<f64>()?),
            2 => assert_eq!(z.to_vec2::<f64>()?, expected.to_vec2::<f64>()?),
            3 => assert_eq!(z.to_vec3::<f64>()?, expected.to_vec3::<f64>()?),
            _ => unreachable!(),
        };

        Ok(())
    }

    Ok(())
}

// "Less"
#[test]
fn test_less() -> Result<()> {
    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-81
    test(&[1., 2., 3.], &[3., 2., 1.], &[1u8, 0, 0])?;

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-81
    test(&[1., 2., 3.], 2., &[1u8, 0, 0])?;

    fn test(a: impl NdArray, b: impl NdArray, expected: impl NdArray) -> Result<()> {
        let manual_graph = create_model_proto_with_graph(Some(GraphProto {
            node: vec![NodeProto {
                op_type: "Less".to_string(),
                domain: "".to_string(),
                attribute: vec![],
                input: vec![INPUT_X.to_string(), INPUT_Y.to_string()],
                output: vec![OUTPUT_Z.to_string()],
                name: "".to_string(),
                doc_string: "".to_string(),
            }],
            name: "".to_string(),
            initializer: vec![],
            input: vec![],
            output: vec![ValueInfoProto {
                name: OUTPUT_Z.to_string(),
                doc_string: "".to_string(),
                r#type: None,
            }],
            value_info: vec![],
            doc_string: "".to_string(),
            sparse_initializer: vec![],
            quantization_annotation: vec![],
        }));

        let mut inputs: HashMap<String, Tensor> = HashMap::new();
        inputs.insert(INPUT_X.to_string(), Tensor::new(a, &Device::Cpu)?);
        inputs.insert(INPUT_Y.to_string(), Tensor::new(b, &Device::Cpu)?);

        let eval = candle_onnx::simple_eval(&manual_graph, inputs)?;
        assert_eq!(eval.len(), 1);

        let z = eval
            .get(OUTPUT_Z)
            .expect("Output 'z' not found")
            .to_dtype(DType::F64)?;

        let expected = Tensor::new(expected, &Device::Cpu)?.to_dtype(DType::F64)?;
        match expected.dims().len() {
            0 => assert_eq!(z.to_vec0::<f64>()?, expected.to_vec0::<f64>()?),
            1 => assert_eq!(z.to_vec1::<f64>()?, expected.to_vec1::<f64>()?),
            2 => assert_eq!(z.to_vec2::<f64>()?, expected.to_vec2::<f64>()?),
            3 => assert_eq!(z.to_vec3::<f64>()?, expected.to_vec3::<f64>()?),
            _ => unreachable!(),
        };

        Ok(())
    }

    Ok(())
}

// "Log"
#[test]
fn test_log() -> Result<()> {
    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-82
    test(&[1., 10.], &[0., std::f64::consts::LN_10])?;

    fn test(data: impl NdArray, expected: impl NdArray) -> Result<()> {
        let manual_graph = create_model_proto_with_graph(Some(GraphProto {
            node: vec![NodeProto {
                op_type: "Log".to_string(),
                domain: "".to_string(),
                attribute: vec![],
                input: vec![INPUT_X.to_string()],
                output: vec![OUTPUT_Z.to_string()],
                name: "".to_string(),
                doc_string: "".to_string(),
            }],
            name: "".to_string(),
            initializer: vec![],
            input: vec![],
            output: vec![ValueInfoProto {
                name: OUTPUT_Z.to_string(),
                doc_string: "".to_string(),
                r#type: None,
            }],
            value_info: vec![],
            doc_string: "".to_string(),
            sparse_initializer: vec![],
            quantization_annotation: vec![],
        }));

        let mut inputs: HashMap<String, Tensor> = HashMap::new();
        inputs.insert(INPUT_X.to_string(), Tensor::new(data, &Device::Cpu)?);

        let eval = candle_onnx::simple_eval(&manual_graph, inputs)?;
        assert_eq!(eval.len(), 1);

        let z = eval.get(OUTPUT_Z).expect("Output 'z' not found");

        let expected = Tensor::new(expected, &Device::Cpu)?;
        match expected.dims().len() {
            0 => assert_eq!(z.to_vec0::<f64>()?, expected.to_vec0::<f64>()?),
            1 => assert_eq!(z.to_vec1::<f64>()?, expected.to_vec1::<f64>()?),
            2 => assert_eq!(z.to_vec2::<f64>()?, expected.to_vec2::<f64>()?),
            3 => assert_eq!(z.to_vec3::<f64>()?, expected.to_vec3::<f64>()?),
            _ => unreachable!(),
        };

        Ok(())
    }

    Ok(())
}

// "Min"
#[test]
fn test_min() -> Result<()> {
    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-94
    test(&[3., 2., 1.], &[1., 4., 4.], &[2., 5., 0.], &[1., 2., 0.])?;

    fn test(
        a: impl NdArray,
        b: impl NdArray,
        c: impl NdArray,
        expected: impl NdArray,
    ) -> Result<()> {
        let manual_graph = create_model_proto_with_graph(Some(GraphProto {
            node: vec![NodeProto {
                op_type: "Min".to_string(),
                domain: "".to_string(),
                attribute: vec![],
                input: vec![
                    INPUT_X.to_string(),
                    INPUT_Y.to_string(),
                    INPUT_A.to_string(),
                ],
                output: vec![OUTPUT_Z.to_string()],
                name: "".to_string(),
                doc_string: "".to_string(),
            }],
            name: "".to_string(),
            initializer: vec![],
            input: vec![],
            output: vec![ValueInfoProto {
                name: OUTPUT_Z.to_string(),
                doc_string: "".to_string(),
                r#type: None,
            }],
            value_info: vec![],
            doc_string: "".to_string(),
            sparse_initializer: vec![],
            quantization_annotation: vec![],
        }));

        let mut inputs: HashMap<String, Tensor> = HashMap::new();
        inputs.insert(INPUT_X.to_string(), Tensor::new(a, &Device::Cpu)?);
        inputs.insert(INPUT_Y.to_string(), Tensor::new(b, &Device::Cpu)?);
        inputs.insert(INPUT_A.to_string(), Tensor::new(c, &Device::Cpu)?);

        let eval = candle_onnx::simple_eval(&manual_graph, inputs)?;
        assert_eq!(eval.len(), 1);

        let z = eval.get(OUTPUT_Z).expect("Output 'z' not found");

        let expected = Tensor::new(expected, &Device::Cpu)?;
        match expected.dims().len() {
            0 => assert_eq!(z.to_vec0::<f64>()?, expected.to_vec0::<f64>()?),
            1 => assert_eq!(z.to_vec1::<f64>()?, expected.to_vec1::<f64>()?),
            2 => assert_eq!(z.to_vec2::<f64>()?, expected.to_vec2::<f64>()?),
            3 => assert_eq!(z.to_vec3::<f64>()?, expected.to_vec3::<f64>()?),
            _ => unreachable!(),
        };

        Ok(())
    }

    Ok(())
}

// "Where"
#[test]
fn test_where() -> Result<()> {
    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-173
    test(
        &[[1u8, 0], [1, 1]],
        &[[1i64, 2], [3, 4]],
        &[[9i64, 8], [7, 6]],
        &[[1i64, 8], [3, 4]],
    )?;

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-173
    test(
        &[[1u8, 0], [1, 1]],
        &[[1., 2.], [3., 4.]],
        &[[9., 8.], [7., 6.]],
        &[[1., 8.], [3., 4.]],
    )?;

    fn test(
        condition: impl NdArray,
        x: impl NdArray,
        y: impl NdArray,
        expected: impl NdArray,
    ) -> Result<()> {
        let manual_graph = create_model_proto_with_graph(Some(GraphProto {
            node: vec![NodeProto {
                op_type: "Where".to_string(),
                domain: "".to_string(),
                attribute: vec![],
                input: vec![
                    INPUT_X.to_string(),
                    INPUT_Y.to_string(),
                    INPUT_A.to_string(),
                ],
                output: vec![OUTPUT_Z.to_string()],
                name: "".to_string(),
                doc_string: "".to_string(),
            }],
            name: "".to_string(),
            initializer: vec![],
            input: vec![],
            output: vec![ValueInfoProto {
                name: OUTPUT_Z.to_string(),
                doc_string: "".to_string(),
                r#type: None,
            }],
            value_info: vec![],
            doc_string: "".to_string(),
            sparse_initializer: vec![],
            quantization_annotation: vec![],
        }));

        let mut inputs: HashMap<String, Tensor> = HashMap::new();
        inputs.insert(INPUT_X.to_string(), Tensor::new(condition, &Device::Cpu)?);
        inputs.insert(INPUT_Y.to_string(), Tensor::new(x, &Device::Cpu)?);
        inputs.insert(INPUT_A.to_string(), Tensor::new(y, &Device::Cpu)?);

        let eval = candle_onnx::simple_eval(&manual_graph, inputs)?;
        assert_eq!(eval.len(), 1);

        let z = eval
            .get(OUTPUT_Z)
            .expect("Output 'z' not found")
            .to_dtype(DType::F64)?;

        let expected = Tensor::new(expected, &Device::Cpu)?.to_dtype(DType::F64)?;
        match expected.dims().len() {
            0 => assert_eq!(z.to_vec0::<f64>()?, expected.to_vec0::<f64>()?),
            1 => assert_eq!(z.to_vec1::<f64>()?, expected.to_vec1::<f64>()?),
            2 => assert_eq!(z.to_vec2::<f64>()?, expected.to_vec2::<f64>()?),
            3 => assert_eq!(z.to_vec3::<f64>()?, expected.to_vec3::<f64>()?),
            _ => unreachable!(),
        };

        Ok(())
    }

    Ok(())
}

#[test]
fn test_floor() -> Result<()> {
    let manual_graph = create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: "Floor".to_string(),
            domain: "".to_string(),
            attribute: vec![],
            input: vec![INPUT_X.to_string()],
            output: vec![OUTPUT_Z.to_string()],
            name: "".to_string(),
            doc_string: "".to_string(),
        }],
        name: "".to_string(),
        initializer: vec![],
        input: vec![ValueInfoProto {
            name: INPUT_X.to_string(),
            doc_string: "".to_string(),
            r#type: None,
        }],
        output: vec![ValueInfoProto {
            name: OUTPUT_Z.to_string(),
            doc_string: "".to_string(),
            r#type: None,
        }],
        value_info: vec![],
        doc_string: "".to_string(),
        sparse_initializer: vec![],
        quantization_annotation: vec![],
    }));
    let x = Tensor::from_vec(
        // some values taken from https://numpy.org/doc/stable/reference/generated/numpy.floor.html
        vec![
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            -1.7,
            -1.5,
            -0.2,
            0.2,
            1.5,
            1.7,
            2.0,
        ],
        &[10],
        &Device::Cpu,
    )?;

    let mut inputs: HashMap<String, Tensor> = HashMap::new();
    inputs.insert(INPUT_X.to_string(), x);

    let eval = candle_onnx::simple_eval(&manual_graph, inputs)?;
    assert_eq!(eval.len(), 1);

    let z = eval.get(OUTPUT_Z).expect("Output 'z' not found");

    let results = z.to_vec1::<f64>()?;

    assert!(results[0].is_nan());
    assert_eq!(
        results[1..],
        vec![
            f64::INFINITY,
            f64::NEG_INFINITY,
            -2.,
            -2.,
            -1.,
            0.,
            1.,
            1.,
            2.
        ]
    );

    Ok(())
}

#[test]
fn test_ceil() -> Result<()> {
    let manual_graph = create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: "Ceil".to_string(),
            domain: "".to_string(),
            attribute: vec![],
            input: vec![INPUT_X.to_string()],
            output: vec![OUTPUT_Z.to_string()],
            name: "".to_string(),
            doc_string: "".to_string(),
        }],
        name: "".to_string(),
        initializer: vec![],
        input: vec![ValueInfoProto {
            name: INPUT_X.to_string(),
            doc_string: "".to_string(),
            r#type: None,
        }],
        output: vec![ValueInfoProto {
            name: OUTPUT_Z.to_string(),
            doc_string: "".to_string(),
            r#type: None,
        }],
        value_info: vec![],
        doc_string: "".to_string(),
        sparse_initializer: vec![],
        quantization_annotation: vec![],
    }));
    let x = Tensor::from_vec(
        // some values taken from https://numpy.org/doc/stable/reference/generated/numpy.ceil.html
        vec![
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            -1.7,
            -1.5,
            -0.2,
            0.2,
            1.5,
            1.7,
            2.0,
        ],
        &[10],
        &Device::Cpu,
    )?;

    let mut inputs: HashMap<String, Tensor> = HashMap::new();
    inputs.insert(INPUT_X.to_string(), x);

    let eval = candle_onnx::simple_eval(&manual_graph, inputs)?;
    assert_eq!(eval.len(), 1);

    let z = eval.get(OUTPUT_Z).expect("Output 'z' not found");

    let results = z.to_vec1::<f64>()?;

    assert!(results[0].is_nan());
    assert_eq!(
        results[1..],
        vec![
            f64::INFINITY,
            f64::NEG_INFINITY,
            -1.,
            -1.,
            -0.,
            1.,
            2.,
            2.,
            2.
        ]
    );

    Ok(())
}

// "ArgMin"
#[test]
fn test_argmin() -> Result<()> {
    // tests from https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-7
    // default_axes_keepdims
    test(
        &[[2u32, 1u32], [3u32, 10u32]],
        None,
        Some(1),
        None,
        &[[0i64, 0i64]],
    )?;
    // keepdims
    test(
        &[[2u32, 1u32], [3u32, 10u32]],
        Some(1),
        Some(1),
        None,
        &[[1i64], [0i64]],
    )?;
    // // negative_axis_keepdims
    test(
        &[[2u32, 1u32], [3u32, 10u32]],
        Some(-1),
        Some(1),
        None,
        &[[1i64], [0i64]],
    )?;
    // no_keepdims
    test(
        &[[2u32, 1u32], [3u32, 10u32]],
        None,
        Some(0),
        None,
        &[0i64, 0i64],
    )?;
    // tests from https://pytorch.org/docs/stable/generated/torch.argmin.html#torch.argmin
    test(
        &[
            [0.1139, 0.2254, -0.1381, 0.3687],
            [1.0100, -1.1975, -0.0102, -0.4732],
            [-0.9240, 0.1207, -0.7506, -1.0213],
            [1.7809, -1.2960, 0.9384, 0.1438],
        ],
        Some(1),
        Some(0),
        None,
        &[2i64, 1i64, 3i64, 1i64],
    )?;
    test(
        &[
            [0.1139, 0.2254, -0.1381, 0.3687],
            [1.0100, -1.1975, -0.0102, -0.4732],
            [-0.9240, 0.1207, -0.7506, -1.0213],
            [1.7809, -1.2960, 0.9384, 0.1438],
        ],
        Some(1),
        None,
        None,
        &[[2i64], [1i64], [3i64], [1i64]],
    )?;
    fn test(
        data: impl NdArray,
        axis: Option<i64>,
        keepdims: Option<i64>,
        select_last_index: Option<i64>,
        expected: impl NdArray,
    ) -> Result<()> {
        let att_axis = AttributeProto {
            name: "axis".to_string(),
            ref_attr_name: "axis".to_string(),
            i: axis.unwrap_or(0),
            doc_string: "axis".to_string(),
            r#type: 2, // INT
            f: 0.0,
            s: vec![],
            t: None,
            g: None,
            sparse_tensor: None,
            tp: None,
            floats: vec![],
            ints: vec![],
            strings: vec![],
            tensors: vec![],
            graphs: vec![],
            sparse_tensors: vec![],
            type_protos: vec![],
        };
        let att_keepdims = AttributeProto {
            name: "keepdims".to_string(),
            ref_attr_name: "keepdims".to_string(),
            i: keepdims.unwrap_or(1),
            doc_string: "keepdims".to_string(),
            r#type: 2, // INT
            f: 0.0,
            s: vec![],
            t: None,
            g: None,
            sparse_tensor: None,
            tp: None,
            floats: vec![],
            ints: vec![],
            strings: vec![],
            tensors: vec![],
            graphs: vec![],
            sparse_tensors: vec![],
            type_protos: vec![],
        };
        let att_select_last_index = AttributeProto {
            name: "select_last_index".to_string(),
            ref_attr_name: "select_last_index".to_string(),
            i: select_last_index.unwrap_or(0),
            doc_string: "select_last_index".to_string(),
            r#type: 2, // INT
            f: 0.0,
            s: vec![],
            t: None,
            g: None,
            sparse_tensor: None,
            tp: None,
            floats: vec![],
            ints: vec![],
            strings: vec![],
            tensors: vec![],
            graphs: vec![],
            sparse_tensors: vec![],
            type_protos: vec![],
        };
        let attrs = {
            let mut mut_attrs = vec![];
            if axis.is_some() {
                mut_attrs.push(att_axis);
            }
            if keepdims.is_some() {
                mut_attrs.push(att_keepdims);
            }
            if select_last_index.is_some() {
                mut_attrs.push(att_select_last_index);
            }
            mut_attrs
        };
        let manual_graph = create_model_proto_with_graph(Some(GraphProto {
            node: vec![NodeProto {
                op_type: "ArgMin".to_string(),
                domain: "".to_string(),
                attribute: attrs,
                input: vec![INPUT_X.to_string()],
                output: vec![OUTPUT_Z.to_string()],
                name: "".to_string(),
                doc_string: "".to_string(),
            }],
            name: "".to_string(),
            initializer: vec![],
            input: vec![],
            output: vec![ValueInfoProto {
                name: OUTPUT_Z.to_string(),
                doc_string: "".to_string(),
                r#type: None,
            }],
            value_info: vec![],
            doc_string: "".to_string(),
            sparse_initializer: vec![],
            quantization_annotation: vec![],
        }));
        let mut inputs: HashMap<String, Tensor> = HashMap::new();
        inputs.insert(INPUT_X.to_string(), Tensor::new(data, &Device::Cpu)?);
        let eval = candle_onnx::simple_eval(&manual_graph, inputs)?;
        let z = eval.get(OUTPUT_Z).expect("Output 'z' not found");

        let expected = Tensor::new(expected, &Device::Cpu)?;
        match expected.dims().len() {
            1 => assert_eq!(z.to_vec1::<i64>()?, expected.to_vec1::<i64>()?),
            2 => assert_eq!(z.to_vec2::<i64>()?, expected.to_vec2::<i64>()?),
            _ => unreachable!(),
        };

        Ok(())
    }

    Ok(())
}

// "ArgMax"
#[test]
fn test_argmax() -> Result<()> {
    // tests from https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-6
    // default_axes_keepdims
    test(
        &[[2u32, 1u32], [3u32, 10u32]],
        None,
        Some(1),
        None,
        &[[1i64, 1i64]],
    )?;
    // keepdims
    test(
        &[[2u32, 1u32], [3u32, 10u32]],
        Some(1),
        Some(1),
        None,
        &[[0i64], [1i64]],
    )?;
    // // negative_axis_keepdims
    test(
        &[[2u32, 1u32], [3u32, 10u32]],
        Some(-1),
        Some(1),
        None,
        &[[0i64], [1i64]],
    )?;
    // no_keepdims
    test(
        &[[2u32, 1u32], [3u32, 10u32]],
        None,
        Some(0),
        None,
        &[1i64, 1i64],
    )?;
    // tests from https://pytorch.org/docs/stable/generated/torch.argmax.html
    test(
        &[
            [1.3398, 0.2663, -0.2686, 0.2450],
            [-0.7401, -0.8805, -0.3402, -1.1936],
            [0.4907, -1.3948, -1.0691, -0.3132],
            [-1.6092, 0.5419, -0.2993, 0.3195],
        ],
        Some(1),
        Some(0),
        None,
        &[0i64, 2i64, 0i64, 1i64],
    )?;
    test(
        &[
            [1.3398, 0.2663, -0.2686, 0.2450],
            [-0.7401, -0.8805, -0.3402, -1.1936],
            [0.4907, -1.3948, -1.0691, -0.3132],
            [-1.6092, 0.5419, -0.2993, 0.3195],
        ],
        Some(1),
        None,
        None,
        &[[0i64], [2i64], [0i64], [1i64]],
    )?;
    fn test(
        data: impl NdArray,
        axis: Option<i64>,
        keepdims: Option<i64>,
        select_last_index: Option<i64>,
        expected: impl NdArray,
    ) -> Result<()> {
        let att_axis = AttributeProto {
            name: "axis".to_string(),
            ref_attr_name: "axis".to_string(),
            i: axis.unwrap_or(0),
            doc_string: "axis".to_string(),
            r#type: 2, // INT
            f: 0.0,
            s: vec![],
            t: None,
            g: None,
            sparse_tensor: None,
            tp: None,
            floats: vec![],
            ints: vec![],
            strings: vec![],
            tensors: vec![],
            graphs: vec![],
            sparse_tensors: vec![],
            type_protos: vec![],
        };
        let att_keepdims = AttributeProto {
            name: "keepdims".to_string(),
            ref_attr_name: "keepdims".to_string(),
            i: keepdims.unwrap_or(1),
            doc_string: "keepdims".to_string(),
            r#type: 2, // INT
            f: 0.0,
            s: vec![],
            t: None,
            g: None,
            sparse_tensor: None,
            tp: None,
            floats: vec![],
            ints: vec![],
            strings: vec![],
            tensors: vec![],
            graphs: vec![],
            sparse_tensors: vec![],
            type_protos: vec![],
        };
        let att_select_last_index = AttributeProto {
            name: "select_last_index".to_string(),
            ref_attr_name: "select_last_index".to_string(),
            i: select_last_index.unwrap_or(0),
            doc_string: "select_last_index".to_string(),
            r#type: 2, // INT
            f: 0.0,
            s: vec![],
            t: None,
            g: None,
            sparse_tensor: None,
            tp: None,
            floats: vec![],
            ints: vec![],
            strings: vec![],
            tensors: vec![],
            graphs: vec![],
            sparse_tensors: vec![],
            type_protos: vec![],
        };
        let attrs = {
            let mut mut_attrs = vec![];
            if axis.is_some() {
                mut_attrs.push(att_axis);
            }
            if keepdims.is_some() {
                mut_attrs.push(att_keepdims);
            }
            if select_last_index.is_some() {
                mut_attrs.push(att_select_last_index);
            }
            mut_attrs
        };
        let manual_graph = create_model_proto_with_graph(Some(GraphProto {
            node: vec![NodeProto {
                op_type: "ArgMax".to_string(),
                domain: "".to_string(),
                attribute: attrs,
                input: vec![INPUT_X.to_string()],
                output: vec![OUTPUT_Z.to_string()],
                name: "".to_string(),
                doc_string: "".to_string(),
            }],
            name: "".to_string(),
            initializer: vec![],
            input: vec![],
            output: vec![ValueInfoProto {
                name: OUTPUT_Z.to_string(),
                doc_string: "".to_string(),
                r#type: None,
            }],
            value_info: vec![],
            doc_string: "".to_string(),
            sparse_initializer: vec![],
            quantization_annotation: vec![],
        }));
        let mut inputs: HashMap<String, Tensor> = HashMap::new();
        inputs.insert(INPUT_X.to_string(), Tensor::new(data, &Device::Cpu)?);
        let eval = candle_onnx::simple_eval(&manual_graph, inputs)?;
        let z = eval.get(OUTPUT_Z).expect("Output 'z' not found");

        let expected = Tensor::new(expected, &Device::Cpu)?;
        match expected.dims().len() {
            1 => assert_eq!(z.to_vec1::<i64>()?, expected.to_vec1::<i64>()?),
            2 => assert_eq!(z.to_vec2::<i64>()?, expected.to_vec2::<i64>()?),
            _ => unreachable!(),
        };

        Ok(())
    }

    Ok(())
}

// "LeakyRelu"
#[test]
fn test_leakyrelu() -> Result<()> {
    // tests from https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-80
    // leakyrelu
    test(&[-1.0, 0.0, 1.0], Some(0.1), &[-0.1, 0.0, 1.0])?;
    fn test(data: impl NdArray, alpha: Option<f32>, expected: impl NdArray) -> Result<()> {
        let att_alpha = AttributeProto {
            name: "alpha".to_string(),
            ref_attr_name: "alpha".to_string(),
            i: 0,
            doc_string: "alpha".to_string(),
            r#type: 1, // FLOAT
            f: alpha.unwrap_or(0.01),
            s: vec![],
            t: None,
            g: None,
            sparse_tensor: None,
            tp: None,
            floats: vec![],
            ints: vec![],
            strings: vec![],
            tensors: vec![],
            graphs: vec![],
            sparse_tensors: vec![],
            type_protos: vec![],
        };
        let attrs = {
            let mut mut_attrs = vec![];
            if alpha.is_some() {
                mut_attrs.push(att_alpha);
            }
            mut_attrs
        };
        let manual_graph = create_model_proto_with_graph(Some(GraphProto {
            node: vec![NodeProto {
                op_type: "LeakyRelu".to_string(),
                domain: "".to_string(),
                attribute: attrs,
                input: vec![INPUT_X.to_string()],
                output: vec![OUTPUT_Z.to_string()],
                name: "".to_string(),
                doc_string: "".to_string(),
            }],
            name: "".to_string(),
            initializer: vec![],
            input: vec![],
            output: vec![ValueInfoProto {
                name: OUTPUT_Z.to_string(),
                doc_string: "".to_string(),
                r#type: None,
            }],
            value_info: vec![],
            doc_string: "".to_string(),
            sparse_initializer: vec![],
            quantization_annotation: vec![],
        }));
        let mut inputs: HashMap<String, Tensor> = HashMap::new();
        inputs.insert(INPUT_X.to_string(), Tensor::new(data, &Device::Cpu)?);
        let eval = candle_onnx::simple_eval(&manual_graph, inputs)?;
        let z = eval.get(OUTPUT_Z).expect("Output 'z' not found");

        let expected = Tensor::new(expected, &Device::Cpu)?;
        for both in z
            .to_vec1::<f64>()?
            .iter()
            .zip(expected.to_vec1::<f64>()?.iter())
        {
            let (act, exp) = both;
            assert!(f64::abs(act - exp) < f32::EPSILON.into());
        }

        Ok(())
    }

    Ok(())
}

// "If"
#[test]
fn test_if() -> Result<()> {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![5.0, 4.0, 3.0, 2.0, 1.0];
    let output_type_proto = Some(TypeProto {
        value: Some(type_proto::Value::TensorType(type_proto::Tensor {
            elem_type: DataType::Float.into(),
            shape: Some(TensorShapeProto {
                dim: vec![Dimension {
                    denotation: "".to_string(),
                    value: Some(dimension::Value::DimValue(5)),
                }],
            }),
        })),
        denotation: "".to_string(),
    });
    let then_branch = GraphProto {
        output: vec![ValueInfoProto {
            name: "then_out".to_string(),
            r#type: output_type_proto.clone(),
            doc_string: "".to_string(),
        }],
        node: vec![NodeProto {
            op_type: "Constant".to_string(),
            input: vec![],
            output: vec!["then_out".to_string()],
            attribute: vec![AttributeProto {
                name: "value".to_string(),
                r#type: AttributeType::Tensor.into(),
                t: Some(TensorProto {
                    dims: vec![x.len() as i64],
                    float_data: x.clone(),
                    data_type: DataType::Float.into(),
                    ..TensorProto::default()
                }),
                ..AttributeProto::default()
            }],
            ..NodeProto::default()
        }],
        ..GraphProto::default()
    };
    let else_branch = GraphProto {
        output: vec![ValueInfoProto {
            name: "else_out".to_string(),
            r#type: output_type_proto.clone(),
            doc_string: "".to_string(),
        }],
        node: vec![NodeProto {
            op_type: "Constant".to_string(),
            input: vec![],
            output: vec!["else_out".to_string()],
            attribute: vec![AttributeProto {
                name: "value".to_string(),
                r#type: AttributeType::Tensor.into(),
                t: Some(TensorProto {
                    dims: vec![y.len() as i64],
                    float_data: y.clone(),
                    data_type: DataType::Float.into(),
                    ..TensorProto::default()
                }),
                ..AttributeProto::default()
            }],
            ..NodeProto::default()
        }],
        ..GraphProto::default()
    };
    let manual_graph = create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: "If".to_string(),
            attribute: vec![
                AttributeProto {
                    name: "then_branch".to_string(),
                    r#type: AttributeType::Graph.into(),
                    g: Some(then_branch),
                    ..AttributeProto::default()
                },
                AttributeProto {
                    name: "else_branch".to_string(),
                    r#type: AttributeType::Graph.into(),
                    g: Some(else_branch),
                    ..AttributeProto::default()
                },
            ],
            input: vec!["cond".to_string()],
            output: vec!["res".to_string()],
            ..NodeProto::default()
        }],
        input: vec![],
        output: vec![ValueInfoProto {
            name: "res".to_string(),
            doc_string: "".to_string(),
            r#type: output_type_proto.clone(),
        }],
        ..GraphProto::default()
    }));

    for cond in [1u8, 0] {
        let inputs =
            HashMap::from_iter([("cond".to_string(), Tensor::full(cond, (1,), &Device::Cpu)?)]);
        let outputs = candle_onnx::simple_eval(&manual_graph, inputs)?;
        let expected = if cond != 0 { &x } else { &y };
        let Some(res) = outputs.get("res") else {
            candle::bail!("outputs didn't contain expected key `res`: {outputs:?}");
        };
        assert_eq!(&res.to_vec1::<f32>()?, expected);
    }
    Ok(())
}

#[test]
fn test_pad() -> Result<()> {
    let data = Tensor::from_vec(vec![1.0, 1.2, 2.3, 3.4, 4.5, 5.7], (3, 2), &Device::Cpu)?;
    let pads = Tensor::from_vec(vec![0i64, 2, 0, 0], (4,), &Device::Cpu)?;
    let mode = "reflect";

    let expected = Tensor::from_vec(
        vec![1.0, 1.2, 1.0, 1.2, 2.3, 3.4, 2.3, 3.4, 4.5, 5.7, 4.5, 5.7],
        (3, 4),
        &Device::Cpu,
    )?;

    let model = create_model_proto_with_graph(Some(GraphProto {
        input: vec![
            ValueInfoProto {
                name: "data".to_string(),
                ..ValueInfoProto::default()
            },
            ValueInfoProto {
                name: "pads".to_string(),
                ..ValueInfoProto::default()
            },
        ],
        output: vec![ValueInfoProto {
            name: "output".to_string(),
            ..ValueInfoProto::default()
        }],
        node: vec![NodeProto {
            op_type: "Pad".to_string(),
            input: vec!["data".to_string(), "pads".to_string()],
            output: vec!["output".to_string()],
            attribute: vec![AttributeProto {
                name: "mode".to_string(),
                r#type: AttributeType::String.into(),
                s: mode.as_bytes().to_vec(),
                ..AttributeProto::default()
            }],
            ..NodeProto::default()
        }],
        ..GraphProto::default()
    }));

    let inputs = HashMap::from_iter([("data".to_string(), data), ("pads".to_string(), pads)]);
    let res = candle_onnx::simple_eval(&model, inputs)?;
    let Some(actual) = res.get("output") else {
        candle::bail!("outputs didn't contain expected key `output`: {res:?}");
    };

    assert_eq!(actual.to_vec2::<f64>()?, expected.to_vec2::<f64>()?);
    Ok(())
}

#[test]
fn test_slice() -> Result<()> {
    let model = create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: "Slice".to_string(),
            input: vec![
                "data".to_string(),
                "starts".to_string(),
                "ends".to_string(),
                "axes".to_string(),
                "steps".to_string(),
            ],
            output: vec!["result".to_string()],
            ..NodeProto::default()
        }],
        input: ["data", "starts", "ends", "axes", "steps"]
            .into_iter()
            .map(|name| ValueInfoProto {
                name: name.to_string(),
                r#type: None,
                doc_string: "".to_string(),
            })
            .collect(),
        output: ["result"]
            .into_iter()
            .map(|name| ValueInfoProto {
                name: name.to_string(),
                r#type: None,
                doc_string: "".to_string(),
            })
            .collect(),
        ..GraphProto::default()
    }));

    /*
    data = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
    ]
    axes = [0, 1]
    starts = [1, 0]
    ends = [2, 3]
    steps = [1, 2]
    result = [
        [5, 7],
    ]
    */

    let outputs = candle_onnx::simple_eval(
        &model,
        HashMap::from_iter([
            (
                "data".to_string(),
                Tensor::from_vec(vec![1i64, 2, 3, 4, 5, 6, 7, 8], (2, 4), &Device::Cpu)?,
            ),
            (
                "starts".to_string(),
                Tensor::from_vec(vec![1i64, 0], (2,), &Device::Cpu)?,
            ),
            (
                "ends".to_string(),
                Tensor::from_vec(vec![2i64, 3], (2,), &Device::Cpu)?,
            ),
            (
                "axes".to_string(),
                Tensor::from_vec(vec![0i64, 1], (2,), &Device::Cpu)?,
            ),
            (
                "steps".to_string(),
                Tensor::from_vec(vec![1i64, 2], (2,), &Device::Cpu)?,
            ),
        ]),
    )?;
    let actual = outputs.get("result").unwrap().to_vec2::<i64>()?;
    assert_eq!(actual, vec![vec![5i64, 7]]);

    /*
    data = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
    ]
    starts = [0, 1]
    ends = [-1, 1000]
    result = [
        [2, 3, 4],
    ]
    */
    let model = create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: "Slice".to_string(),
            input: vec!["data".to_string(), "starts".to_string(), "ends".to_string()],
            output: vec!["result".to_string()],
            ..NodeProto::default()
        }],
        input: ["data", "starts", "ends"]
            .into_iter()
            .map(|name| ValueInfoProto {
                name: name.to_string(),
                r#type: None,
                doc_string: "".to_string(),
            })
            .collect(),
        output: ["result"]
            .into_iter()
            .map(|name| ValueInfoProto {
                name: name.to_string(),
                r#type: None,
                doc_string: "".to_string(),
            })
            .collect(),
        ..GraphProto::default()
    }));
    let outputs = candle_onnx::simple_eval(
        &model,
        HashMap::from_iter([
            (
                "data".to_string(),
                Tensor::from_vec(vec![1i64, 2, 3, 4, 5, 6, 7, 8], (2, 4), &Device::Cpu)?,
            ),
            (
                "starts".to_string(),
                Tensor::from_vec(vec![0i64, 1], (2,), &Device::Cpu)?,
            ),
            (
                "ends".to_string(),
                Tensor::from_vec(vec![-1i64, 1000], (2,), &Device::Cpu)?,
            ),
        ]),
    )?;
    let actual = outputs.get("result").unwrap().to_vec2::<i64>()?;
    assert_eq!(actual, vec![vec![2i64, 3, 4]]);

    Ok(())
}
