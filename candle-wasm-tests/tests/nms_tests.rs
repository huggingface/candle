#![allow(unused_imports, unexpected_cfgs)]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test as test;
#[cfg(not(target_arch = "wasm32"))]
use tokio::test as test;
use candle_wasm_tests::{
    to_vec0_round_async, to_vec1_round_async, to_vec2_round_async, to_vec3_round_async,
};
use candle::Result;
use candle_transformers::object_detection::{
    non_maximum_suppression, soft_non_maximum_suppression, Bbox,
};
#[test]
async fn nms_basic() -> Result<()> {
    let mut bboxes = vec![
        vec![Bbox { xmin : 245.0, ymin : 305.0, xmax : 575.0, ymax : 490.0, confidence :
        0.9, data : (), }, Bbox { xmin : 235.0, ymin : 300.0, xmax : 485.0, ymax : 515.0,
        confidence : 0.8, data : (), }, Bbox { xmin : 305.0, ymin : 270.0, xmax : 540.0,
        ymax : 500.0, confidence : 0.6, data : (), },]
    ];
    non_maximum_suppression(&mut bboxes, 0.5);
    let bboxes = bboxes.into_iter().next().unwrap();
    assert_eq!(bboxes.len(), 1);
    assert_eq!(bboxes[0].confidence, 0.9);
    Ok(())
}
#[test]
async fn softnms_basic_functionality() -> Result<()> {
    let mut bboxes = vec![
        vec![Bbox { xmin : 0.0, ymin : 0.0, xmax : 1.0, ymax : 1.0, confidence : 0.5,
        data : (), }, Bbox { xmin : 0.1, ymin : 0.1, xmax : 1.1, ymax : 1.1, confidence :
        0.9, data : (), }, Bbox { xmin : 0.2, ymin : 0.2, xmax : 1.2, ymax : 1.2,
        confidence : 0.6, data : (), },]
    ];
    soft_non_maximum_suppression(&mut bboxes, Some(0.5), Some(0.1), Some(0.5));
    assert!(bboxes[0] [0].confidence == 0.9);
    assert!(bboxes[0] [1].confidence < 0.5);
    assert!(bboxes[0] [2].confidence < 0.6);
    Ok(())
}
#[test]
async fn softnms_confidence_decay() -> Result<()> {
    let mut bboxes = vec![
        vec![Bbox { xmin : 0.0, ymin : 0.0, xmax : 1.0, ymax : 1.0, confidence : 0.9,
        data : (), }, Bbox { xmin : 0.1, ymin : 0.1, xmax : 1.1, ymax : 1.1, confidence :
        0.8, data : (), },]
    ];
    soft_non_maximum_suppression(&mut bboxes, Some(0.5), Some(0.1), Some(0.5));
    assert!(bboxes[0] [0].confidence == 0.9);
    assert!(bboxes[0] [1].confidence < 0.8);
    Ok(())
}
#[test]
async fn softnms_confidence_threshold() -> Result<()> {
    let mut bboxes = vec![
        vec![Bbox { xmin : 0.0, ymin : 0.0, xmax : 1.0, ymax : 1.0, confidence : 0.9,
        data : (), }, Bbox { xmin : 0.1, ymin : 0.1, xmax : 1.1, ymax : 1.1, confidence :
        0.05, data : (), },]
    ];
    soft_non_maximum_suppression(&mut bboxes, Some(0.5), Some(0.1), Some(0.5));
    assert_eq!(bboxes[0].len(), 2);
    assert_eq!(bboxes[0] [0].confidence, 0.9);
    assert_eq!(bboxes[0] [1].confidence, 0.00);
    Ok(())
}
#[test]
async fn softnms_no_overlap() -> Result<()> {
    let mut bboxes = vec![
        vec![Bbox { xmin : 0.0, ymin : 0.0, xmax : 1.0, ymax : 1.0, confidence : 0.9,
        data : (), }, Bbox { xmin : 2.0, ymin : 2.0, xmax : 3.0, ymax : 3.0, confidence :
        0.8, data : (), },]
    ];
    soft_non_maximum_suppression(&mut bboxes, Some(0.5), Some(0.1), Some(0.5));
    assert_eq!(bboxes[0].len(), 2);
    assert_eq!(bboxes[0] [0].confidence, 0.9);
    assert_eq!(bboxes[0] [1].confidence, 0.8);
    Ok(())
}
#[test]
async fn softnms_no_bbox() -> Result<()> {
    let mut bboxes: Vec<Vec<Bbox<()>>> = vec![];
    soft_non_maximum_suppression(&mut bboxes, Some(0.5), Some(0.1), Some(0.5));
    assert!(bboxes.is_empty());
    Ok(())
}
#[test]
async fn softnms_single_bbox() -> Result<()> {
    let mut bboxes = vec![
        vec![Bbox { xmin : 0.0, ymin : 0.0, xmax : 1.0, ymax : 1.0, confidence : 0.9,
        data : (), }]
    ];
    soft_non_maximum_suppression(&mut bboxes, Some(0.5), Some(0.1), Some(0.5));
    assert_eq!(bboxes[0].len(), 1);
    Ok(())
}
#[test]
async fn softnms_equal_confidence_overlap() -> Result<()> {
    let mut bboxes = vec![
        vec![Bbox { xmin : 0.0, ymin : 0.0, xmax : 1.0, ymax : 1.0, confidence : 0.5,
        data : (), }, Bbox { xmin : 0.1, ymin : 0.1, xmax : 1.1, ymax : 1.1, confidence :
        0.5, data : (), },]
    ];
    soft_non_maximum_suppression(&mut bboxes, Some(0.5), Some(0.1), Some(0.5));
    assert_eq!(bboxes[0].len(), 2);
    assert!(bboxes[0] [0].confidence == 0.5);
    assert!(bboxes[0] [1].confidence < 0.5);
    Ok(())
}
