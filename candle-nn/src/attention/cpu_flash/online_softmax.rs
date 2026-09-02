//! Online softmax step shared by CPU flash-attention kernels.

/// Stream one (score, v_row) into (m, ssum, acc); v_apply adds v_row * w to acc.
#[inline(always)]
pub(crate) fn online_softmax_step(
    score: f32,
    m: &mut f32,
    ssum: &mut f32,
    acc: &mut [f32],
    v_apply: impl FnOnce(&mut [f32], f32),
) {
    if score > *m {
        let scale_old = (*m - score).exp();
        for a in acc.iter_mut() {
            *a *= scale_old;
        }
        *ssum = *ssum * scale_old + 1.0;
        *m = score;
        v_apply(acc, 1.0);
    } else {
        let w = (score - *m).exp();
        v_apply(acc, w);
        *ssum += w;
    }
}
