use candle::{Result};
use candle_transformers::object_detection::{Bbox, non_maximum_suppression, soft_non_maximum_suppression};


 #[test]
fn test_non_maximum_suppression() -> Result<()> {
    // Boxes based upon https://thepythoncode.com/article/non-maximum-suppression-using-opencv-in-python
    let mut bboxes= vec![
        vec![
            Bbox { xmin: 245.0, ymin: 305.0, xmax: 575.0, ymax: 490.0, confidence: 0.9, data: () }, // Box 1
            Bbox { xmin: 235.0, ymin: 300.0, xmax: 485.0, ymax: 515.0, confidence: 0.8, data: () }, // Box 2
            Bbox { xmin: 305.0, ymin: 270.0, xmax: 540.0, ymax: 500.0, confidence: 0.6, data: () }  // Box 3
        ]
    ];

    non_maximum_suppression(&mut bboxes, 0.5);
    let bboxes = bboxes.into_iter().next().unwrap();
    assert_eq!(bboxes.len(), 1);
    assert_eq!(bboxes[0].confidence, 0.9);

    Ok(())
}

#[test]
fn test_soft_non_maximum_suppression() -> Result<()> {
    // Boxes based upon https://thepythoncode.com/article/non-maximum-suppression-using-opencv-in-python
    let mut bboxes= vec![
        vec![
            Bbox { xmin: 245.0, ymin: 305.0, xmax: 575.0, ymax: 490.0, confidence: 0.9, data: () }, // Box 1
            Bbox { xmin: 235.0, ymin: 300.0, xmax: 485.0, ymax: 515.0, confidence: 0.8, data: () }, // Box 2
            Bbox { xmin: 305.0, ymin: 270.0, xmax: 540.0, ymax: 500.0, confidence: 0.6, data: () }  // Box 3
        ]
    ];

    soft_non_maximum_suppression(&mut bboxes, Some(0.5), Some(0.3), Some(0.5));
    let bboxes = bboxes.into_iter().next().unwrap();
    println!("Bboxes: {:?}", bboxes);
    assert_eq!(bboxes.len(), 2);

    Ok(())
}
