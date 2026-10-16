use candle::{DType, Device, Result, Tensor};
use candle_onnx::onnx::attribute_proto::AttributeType;
use candle_onnx::onnx::{
    tensor_shape_proto, type_proto, AttributeProto, GraphProto, ModelProto, NodeProto,
    OperatorSetIdProto, TensorShapeProto, TypeProto, ValueInfoProto,
};
use prost::Message;
use std::collections::HashMap;

fn int_attr(name: &str, value: i64) -> AttributeProto {
    AttributeProto {
        name: name.into(),
        r#type: AttributeType::Int.into(),
        i: value,
        ..Default::default()
    }
}

fn value_info(name: &str, elem_type: i32, shape: &[usize]) -> ValueInfoProto {
    ValueInfoProto {
        name: name.into(),
        r#type: Some(TypeProto {
            value: Some(type_proto::Value::TensorType(type_proto::Tensor {
                elem_type,
                shape: Some(TensorShapeProto {
                    dim: shape
                        .iter()
                        .map(|&dim| tensor_shape_proto::Dimension {
                            value: Some(tensor_shape_proto::dimension::Value::DimValue(dim as i64)),
                            ..Default::default()
                        })
                        .collect(),
                }),
            })),
            ..Default::default()
        }),
        ..Default::default()
    }
}

fn evaluate(
    op: &str,
    opset: i64,
    axes: Option<&[i64]>,
    keepdims: Option<i64>,
    noop: Option<i64>,
    data: Tensor,
    output_shape: &[usize],
) -> Result<Tensor> {
    let mut attribute = Vec::new();
    if let Some(keepdims) = keepdims {
        attribute.push(int_attr("keepdims", keepdims));
    }
    if let Some(noop) = noop {
        // This attribute only exists since opset 18.
        assert!(opset >= 18);
        attribute.push(int_attr("noop_with_empty_axes", noop));
    }
    let mut input = vec!["data".into()];
    let mut graph_input = vec![value_info("data", 1, data.dims())];
    let mut inputs = HashMap::from([("data".into(), data)]);
    if let Some(axes) = axes {
        if opset >= 18 {
            input.push("axes".into());
            graph_input.push(value_info("axes", 7, &[axes.len()]));
            inputs.insert(
                "axes".into(),
                Tensor::from_vec(axes.to_vec(), axes.len(), &Device::Cpu)?,
            );
        } else {
            attribute.push(AttributeProto {
                name: "axes".into(),
                r#type: AttributeType::Ints.into(),
                ints: axes.to_vec(),
                ..Default::default()
            });
        }
    }
    let model = ModelProto {
        ir_version: 8,
        opset_import: vec![OperatorSetIdProto {
            domain: String::new(),
            version: opset,
        }],
        graph: Some(GraphProto {
            node: vec![NodeProto {
                op_type: op.into(),
                input,
                output: vec!["output".into()],
                attribute,
                ..Default::default()
            }],
            input: graph_input,
            output: vec![value_info("output", 1, output_shape)],
            ..Default::default()
        }),
        ..Default::default()
    };
    // Exercise the protobuf model and the real ONNX evaluator, not a reduction helper.
    let model = ModelProto::decode(model.encode_to_vec().as_slice()).unwrap();
    let outputs = candle_onnx::simple_eval(&model, inputs)?;
    Ok(outputs["output"].clone())
}

fn check_matrix(op: &str, opset: i64) -> Result<()> {
    let data = Tensor::new(&[[3f32, -2., 7.], [4., 6., 1.]], &Device::Cpu)?;
    let noops: &[Option<i64>] = if opset >= 18 {
        &[None, Some(0), Some(1)]
    } else {
        &[None]
    };
    let axes_cases: &[Option<&[i64]>] = &[
        None,
        Some(&[]),
        Some(&[0]),
        Some(&[1]),
        Some(&[-1]),
        Some(&[-2]),
        Some(&[1, 0]),
        Some(&[0, -1]),
    ];
    let mut failures = Vec::new();
    let mut count = 0;
    for &axes in axes_cases {
        for keepdims in [None, Some(0), Some(1)] {
            for &noop in noops {
                let empty = axes.is_none_or(|axes| axes.is_empty());
                let keep = keepdims != Some(0);
                let (shape, values): (Vec<usize>, Vec<f32>) = if empty && noop == Some(1) {
                    (vec![2, 3], vec![3., -2., 7., 4., 6., 1.])
                } else if empty || axes.is_some_and(|axes| axes.len() == 2) {
                    (
                        if keep { vec![1, 1] } else { vec![] },
                        vec![if op == "ReduceMax" { 7. } else { -2. }],
                    )
                } else if axes == Some(&[0][..]) || axes == Some(&[-2][..]) {
                    (
                        if keep { vec![1, 3] } else { vec![3] },
                        if op == "ReduceMax" {
                            vec![4., 6., 7.]
                        } else {
                            vec![3., -2., 1.]
                        },
                    )
                } else {
                    (
                        if keep { vec![2, 1] } else { vec![2] },
                        if op == "ReduceMax" {
                            vec![7., 6.]
                        } else {
                            vec![-2., 1.]
                        },
                    )
                };
                let output = evaluate(op, opset, axes, keepdims, noop, data.clone(), &shape)?;
                let actual = output.flatten_all()?.to_vec1::<f32>()?;
                let case =
                    format!("{op} opset={opset} axes={axes:?} keepdims={keepdims:?} noop={noop:?}");
                if output.dims() != shape || actual != values || output.dtype() != DType::F32 {
                    failures.push(format!(
                        "{case}: got {:?} {actual:?}, expected {shape:?} {values:?}",
                        output.dims()
                    ));
                }
                count += 1;
            }
        }
    }
    eprintln!(
        "{op} opset={opset}: {count} cases, {} failures",
        failures.len()
    );
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    Ok(())
}

#[test]
fn reduce_max_opset13_axes() -> Result<()> {
    check_matrix("ReduceMax", 13)
}

#[test]
fn reduce_min_opset13_axes() -> Result<()> {
    check_matrix("ReduceMin", 13)
}

#[test]
fn reduce_max_opset18_axes() -> Result<()> {
    check_matrix("ReduceMax", 18)
}

#[test]
fn reduce_min_opset18_axes() -> Result<()> {
    check_matrix("ReduceMin", 18)
}
