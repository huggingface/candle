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


pub fn soft_non_maximum_suppression<D>(
    bboxes: &mut [Vec<Bbox<D>>],
    iou_threshold: Option<f32>,
    score_threshold: Option<f32>,
    sigma: Option<f32>,
) {
    // Function based on https://arxiv.org/pdf/1704.04503
    let iou_threshold = iou_threshold.unwrap_or(0.5);
    let score_threshold = score_threshold.unwrap_or(0.1);
    let sigma = sigma.unwrap_or(0.5);

    for bboxes_for_class in bboxes.iter_mut() {
        bboxes_for_class.sort_by(|b1, b2| b2.confidence.partial_cmp(&b1.confidence).unwrap());

        let mut to_remove = vec![];
        let mut updated_confidences = vec![0.0; bboxes_for_class.len()]; // Mutable confidence
        // storage 

        let mut current_index = 0;

        while current_index < bboxes_for_class.len() {
            let current_bbox = &bboxes_for_class[current_index];
            let mut index = current_index + 1;

            while index < bboxes_for_class.len() {
                let iou_val = iou(current_bbox, &bboxes_for_class[index]);
                if iou_val > iou_threshold {
                    // Decay calculation from page 4 of: https://arxiv.org/pdf/1704.04503
                    let decay = (-iou_val * iou_val / sigma).exp();
                    let updated_confidence = bboxes_for_class[index].confidence * decay;

                    if updated_confidence < score_threshold {
                        to_remove.push(index);
                    } else {
                        updated_confidences[index] = updated_confidence;
                    }
                }
                index += 1;
            }

            current_index += 1;
        }

        // Update confidences
        for (i, &confidence) in updated_confidences.iter().enumerate() {
            if confidence > 0.0 {
                bboxes_for_class[i].confidence = confidence;
            }
        }

        // Remove boxes with low confidence in reverse order
        to_remove.sort_by(|a, b| b.cmp(a)); // Reverse sort 
        for &index in to_remove.iter() {
            if index < bboxes_for_class.len() {
                bboxes_for_class.remove(index);
            }
        }
    }
}

