//! Bounding Boxes and Intersection
//!
//! This module provides functionality for handling bounding boxes and their manipulation,
//! particularly in the context of object detection. It includes tools for calculating
//! intersection over union (IoU) and non-maximum suppression (NMS).

/// A bounding box around an object.
#[derive(Debug, Clone)]
pub struct Bbox<D> {
    pub xmin: f32,
    pub ymin: f32,
    pub xmax: f32,
    pub ymax: f32,
    pub confidence: f32,
    pub data: D,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KeyPoint {
    pub x: f32,
    pub y: f32,
    pub mask: f32,
}

/// Intersection over union of two bounding boxes.
pub fn iou<D>(b1: &Bbox<D>, b2: &Bbox<D>) -> f32 {
    let b1_area = (b1.xmax - b1.xmin + 1.) * (b1.ymax - b1.ymin + 1.);
    let b2_area = (b2.xmax - b2.xmin + 1.) * (b2.ymax - b2.ymin + 1.);
    let i_xmin = b1.xmin.max(b2.xmin);
    let i_xmax = b1.xmax.min(b2.xmax);
    let i_ymin = b1.ymin.max(b2.ymin);
    let i_ymax = b1.ymax.min(b2.ymax);
    let i_area = (i_xmax - i_xmin + 1.).max(0.) * (i_ymax - i_ymin + 1.).max(0.);
    i_area / (b1_area + b2_area - i_area)
}

pub fn non_maximum_suppression<D>(bboxes: &mut [Vec<Bbox<D>>], threshold: f32) {
    // Perform non-maximum suppression.
    for bboxes_for_class in bboxes.iter_mut() {
        bboxes_for_class.sort_by(|b1, b2| b2.confidence.partial_cmp(&b1.confidence).unwrap());
        let mut current_index = 0;
        for index in 0..bboxes_for_class.len() {
            let mut drop = false;
            for prev_index in 0..current_index {
                let iou = iou(&bboxes_for_class[prev_index], &bboxes_for_class[index]);
                if iou > threshold {
                    drop = true;
                    break;
                }
            }
            if !drop {
                bboxes_for_class.swap(current_index, index);
                current_index += 1;
            }
        }
        bboxes_for_class.truncate(current_index);
    }
}

// Updates confidences starting at highest and comparing subsequent boxes.
fn update_confidences<D>(
    bboxes_for_class: &[Bbox<D>],
    updated_confidences: &mut [f32],
    iou_threshold: f32,
    sigma: f32,
) {
    let len = bboxes_for_class.len();
    for current_index in 0..len {
        let current_bbox = &bboxes_for_class[current_index];
        for index in (current_index + 1)..len {
            let iou_val = iou(current_bbox, &bboxes_for_class[index]);
            if iou_val > iou_threshold {
                // Decay calculation from page 4 of: https://arxiv.org/pdf/1704.04503
                let decay = (-iou_val * iou_val / sigma).exp();
                let updated_confidence = bboxes_for_class[index].confidence * decay;
                updated_confidences[index] = updated_confidence;
            }
        }
    }
}

// Sorts the bounding boxes by confidence and applies soft non-maximum suppression.
// This function is based on the algorithm described in https://arxiv.org/pdf/1704.04503
pub fn soft_non_maximum_suppression<D>(
    bboxes: &mut [Vec<Bbox<D>>],
    iou_threshold: Option<f32>,
    confidence_threshold: Option<f32>,
    sigma: Option<f32>,
) {
    let iou_threshold = iou_threshold.unwrap_or(0.5);
    let confidence_threshold = confidence_threshold.unwrap_or(0.1);
    let sigma = sigma.unwrap_or(0.5);

    for bboxes_for_class in bboxes.iter_mut() {
        // Sort boxes by confidence in descending order
        bboxes_for_class.sort_by(|b1, b2| b2.confidence.partial_cmp(&b1.confidence).unwrap());
        let mut updated_confidences = bboxes_for_class
            .iter()
            .map(|bbox| bbox.confidence)
            .collect::<Vec<_>>();
        update_confidences(
            bboxes_for_class,
            &mut updated_confidences,
            iou_threshold,
            sigma,
        );
        // Update confidences, set to 0.0 if below threshold
        for (i, &confidence) in updated_confidences.iter().enumerate() {
            bboxes_for_class[i].confidence = if confidence < confidence_threshold {
                0.0
            } else {
                confidence
            };
        }
    }
}
