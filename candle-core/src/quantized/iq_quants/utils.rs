use half::f16;

const GROUP_MAX_EPS: f32 = 1e-15;

#[allow(clippy::too_many_arguments)]
pub(super) fn quantize_row_iq4_nl_impl(
    super_block_size: usize,
    block_size: usize,
    x: &[f32],
    dh: &mut f16,
    q4: &mut [u8],
    scales_h: Option<&mut u16>,
    scales_l: Option<&mut [u8]>,
    scales: &mut [f32],
    weight: &mut [f32],
    lbuf: &mut [u8],
    values: &[i8],
    quant_weights: Option<&[f32]>,
    ntry: i32,
) {
    // For safety, confirm the slices have correct lengths:
    let sb_div_2 = super_block_size / 2;
    let sb_div_32 = super_block_size / 32;
    let sb_div_64 = super_block_size / 64;
    assert_eq!(q4.len(), sb_div_2);
    assert_eq!(scales.len(), sb_div_32);
    assert_eq!(lbuf.len(), super_block_size);
    assert_eq!(weight.len(), block_size);

    // 1. compute sigma2
    let mut sigma2 = 0f32;
    for x in x.iter().take(super_block_size) {
        sigma2 += x * x;
    }
    sigma2 *= 2.0 / (super_block_size as f32);

    // 2. zero out q4, set dh to 0
    for qi in q4.iter_mut() {
        *qi = 0;
    }
    *dh = f16::from_f32(0.0);

    // Track the max absolute scale across sub-blocks
    let mut max_scale = 0.0_f32;
    let mut amax_scale = 0.0_f32;

    // For each 32-float block within the 256-float super-block:
    let nblocks = super_block_size / block_size;

    for ib in 0..nblocks {
        let xb = &x[ib * block_size..ib * block_size + block_size];
        let lb = &mut lbuf[ib * block_size..ib * block_size + block_size];

        // If we have external `quant_weights`, fill `weight[j] = quant_weights[j]*sqrt(...)`,
        // else `weight[j] = xb[j]*xb[j]`
        if let Some(qw) = quant_weights {
            let qw_block = &qw[ib * block_size..ib * block_size + block_size];
            for j in 0..block_size {
                let val = xb[j];
                weight[j] = qw_block[j] * (sigma2 + val * val).sqrt();
            }
        } else {
            for j in 0..block_size {
                let val = xb[j];
                weight[j] = val * val;
            }
        }

        // 3. find amax (largest absolute value in block)
        let mut amax = 0.0_f32;
        let mut max_v = 0.0_f32;
        for &xx in xb {
            let ax = xx.abs();
            if ax > amax {
                amax = ax;
                max_v = xx;
            }
        }

        // If amax is extremely small, scale = 0
        if amax < GROUP_MAX_EPS {
            scales[ib] = 0.0;
            continue;
        }

        // 4. initial guess for d
        let sign_factor = if ntry > 0 { -1.0 } else { 1.0 };
        let mut d = sign_factor * max_v / (values[0] as f32);
        let id = 1.0 / d;

        // 5. compute an initial sumqx, sumq2
        let mut sumqx = 0.0_f32;
        let mut sumq2 = 0.0_f32;
        for j in 0..block_size {
            let val = xb[j];
            let al = id * val;
            let l = best_index_int8(values, al);
            lb[j] = l as u8;

            let q = values[l] as f32;
            let w = weight[j];
            sumqx += w * q * val;
            sumq2 += w * q * q;
        }
        d = sumqx / sumq2;
        let mut best = d * sumqx;

        // 6. do extra tries around that initial guess
        for itry in -ntry..=ntry {
            let test_id = (itry as f32 + values[0] as f32) / max_v;
            let mut tmp_sumqx = 0.0_f32;
            let mut tmp_sumq2 = 0.0_f32;
            for j in 0..block_size {
                let val = xb[j];
                let al = test_id * val;
                let l = best_index_int8(values, al);
                let q = values[l] as f32;
                let w = weight[j];
                tmp_sumqx += w * q * val;
                tmp_sumq2 += w * q * q;
            }
            if tmp_sumq2 > 0.0 {
                let maybe_d = tmp_sumqx / tmp_sumq2;
                let maybe_best = maybe_d * tmp_sumqx;
                if maybe_best > best {
                    best = maybe_best;
                    d = maybe_d;
                }
            }
        }

        // 7. record the chosen scale
        scales[ib] = d;
        let abs_d = d.abs();
        if abs_d > amax_scale {
            amax_scale = abs_d;
            max_scale = d;
        }
    }

    // 8. If we have more than one 32-float block in the super-block:
    if nblocks > 1 {
        let scales_h = scales_h.expect("Expected scales_h, nblocks > 1");
        let scales_l = scales_l.expect("Expected scales_l, nblocks > 1");
        assert_eq!(scales_l.len(), sb_div_64);

        // zero scales_h, because we store 2 bits per block in it
        // for nblocks=8, we store them in a single 16-bit value
        *scales_h = 0;
        for sl in scales_l.iter_mut() {
            *sl = 0;
        }

        let d = -max_scale / 32.0;
        *dh = f16::from_f32(d);
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };

        for ib in 0..nblocks {
            // l = nearest_int(id * scales[ib]), clamp to [-32..31]
            let mut l = (id * scales[ib]).round() as i32;
            l = l.clamp(-32, 31);

            // refine block
            let dl = d * (l as f32);
            let idl = if dl != 0.0 { 1.0 / dl } else { 0.0 };

            let xb = &x[ib * block_size..ib * block_size + block_size];
            let lb = &mut lbuf[ib * block_size..ib * block_size + block_size];
            for j in 0..block_size {
                let val = xb[j];
                lb[j] = best_index_int8(values, idl * val) as u8;
            }

            // store l in 4 bits + 4 bits
            let l_offset = (l + 32) as u8; // now in [0..64)
            let l_low = l_offset & 0x0f;
            let l_high = l_offset >> 4;

            // scales_l[ib/2] uses the nibble for this block
            if ib % 2 == 0 {
                scales_l[ib / 2] = l_low;
            } else {
                scales_l[ib / 2] |= l_low << 4;
            }
            // scales_h for each block (2 bits per block) => stored in a 16-bit
            // scaled_h[ib/8] with (l_high << (2*(ib%8)))
            let shift = 2 * (ib % 8);
            *scales_h |= (l_high as u16) << shift;
        }
    } else {
        // single 32-float block => just store d
        *dh = f16::from_f32(scales[0]);
        if ntry > 0 {
            let id = if scales[0] != 0.0 {
                1.0 / scales[0]
            } else {
                0.0
            };
            for j in 0..super_block_size {
                lbuf[j] = best_index_int8(values, id * x[j]) as u8;
            }
        }
    }

    // 9. Finally, pack all 4-bit values from L into q4
    //    q4[16*i + j] = L[32*i + j] | (L[32*i + 16 + j] << 4)
    for i in 0..(super_block_size / 32) {
        for j in 0..16 {
            let lo = lbuf[32 * i + j] & 0x0f;
            let hi = (lbuf[32 * i + 16 + j] & 0x0f) << 4;
            q4[16 * i + j] = lo | hi;
        }
    }
}

/// Finds the best index i in [0..values.len()) such that
/// `values[i]` is closest to `x`. The array `values` is strictly
/// ascending/
fn best_index_int8(values: &[i8], x: f32) -> usize {
    // Quick boundary checks
    if x <= values[0] as f32 {
        return 0;
    }
    let n = values.len();
    let last = (n - 1).max(0);
    if x >= values[last] as f32 {
        return last;
    }

    // Binary search
    let mut ml = 0;
    let mut mu = last;
    while mu - ml > 1 {
        let mav = (ml + mu) / 2;
        if x < values[mav] as f32 {
            mu = mav;
        } else {
            ml = mav;
        }
    }

    // Return whichever is closer among values[mu-1], values[mu]
    // But watch out if mu == 0 or mu == n-1 ...
    // (the boundary checks above should keep mu>0)
    let dist_left = (x - values[ml] as f32).abs();
    let dist_right = (values[mu] as f32 - x).abs();
    if dist_left <= dist_right {
        ml
    } else {
        mu
    }
}
