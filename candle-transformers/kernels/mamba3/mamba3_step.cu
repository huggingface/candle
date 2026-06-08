#include "mamba3_common.cuh"

extern "C" __global__ void mamba3_siso_step_f32(
    float* __restrict__ out,
    float* __restrict__ out_angle_state,
    float* __restrict__ out_ssm_state,
    float* __restrict__ out_k_state,
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ adt,
    const float* __restrict__ dt,
    const float* __restrict__ trap,
    const float* __restrict__ q_bias,
    const float* __restrict__ k_bias,
    const float* __restrict__ angles,
    const float* __restrict__ d,
    const float* __restrict__ z,
    const float* __restrict__ angle_state,
    const float* __restrict__ ssm_state,
    const float* __restrict__ k_state,
    const float* __restrict__ v_state,
    const uint32_t batch,
    const uint32_t nheads,
    const uint32_t headdim_qk,
    const uint32_t headdim_v,
    const uint32_t headdim_angles,
    const uint32_t has_d,
    const uint32_t has_z
) {
    const uint32_t head = blockIdx.x;
    const uint32_t b = blockIdx.y;
    if (head >= nheads || b >= batch) return;

    const uint32_t qk = headdim_qk;
    const uint32_t vdim = headdim_v;
    const uint32_t base = (b * nheads + head);

    float q_local[128];
    float k_local[128];
    float angles_local[64];

    for (uint32_t i = 0; i < qk; ++i) {
        q_local[i] = q[base * qk + i] + q_bias[head * qk + i];
        k_local[i] = k[base * qk + i] + k_bias[head * qk + i];
    }

    float dt_val = dt[base];
    for (uint32_t i = 0; i < headdim_angles; ++i) {
        float a = angle_state[base * headdim_angles + i];
        a += mamba3_tanh(angles[base * headdim_angles + i]) * 3.141592653589793f * dt_val;
        angles_local[i] = a;
        out_angle_state[base * headdim_angles + i] = a;
    }

    float q_rot[128];
    float k_rot[128];
    mamba3_rope_pairwise(q_local, q_rot, angles_local, qk);
    mamba3_rope_pairwise(k_local, k_rot, angles_local, qk);

    for (uint32_t i = 0; i < qk; ++i) {
        out_k_state[base * qk + i] = k_rot[i];
    }

    float adt_val = adt[base];
    float trap_val = mamba3_sigmoid(trap[base]);
    float alpha = expf(adt_val);
    float beta = alpha * dt_val * (1.f - trap_val);
    float gamma = trap_val * dt_val;

    const uint32_t state_base = (b * nheads + head) * vdim * qk;
    const uint32_t kprev_base = base * qk;
    const uint32_t vprev_base = base * vdim;
    const uint32_t v_base = base * vdim;

    for (uint32_t vi = 0; vi < vdim; ++vi) {
        float v_prev = v_state[vprev_base + vi];
        float v_cur = v[v_base + vi];
        for (uint32_t ki = 0; ki < qk; ++ki) {
            uint32_t idx = state_base + vi * qk + ki;
            float h = ssm_state[idx] * alpha;
            h += beta * v_prev * k_state[kprev_base + ki];
            h += gamma * v_cur * k_rot[ki];
            out_ssm_state[idx] = h;
        }
    }

    for (uint32_t vi = 0; vi < vdim; ++vi) {
        float acc = 0.f;
        for (uint32_t ki = 0; ki < qk; ++ki) {
            acc += out_ssm_state[state_base + vi * qk + ki] * q_rot[ki];
        }
        if (has_d) acc += d[head] * v[v_base + vi];
        if (has_z) acc *= mamba3_silu(z[v_base + vi]);
        out[base * vdim + vi] = acc;
    }
}

extern "C" __global__ void mamba3_mimo_step_f32(
    float* __restrict__ out,
    float* __restrict__ out_angle_state,
    float* __restrict__ out_ssm_state,
    float* __restrict__ out_k_state,
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ adt,
    const float* __restrict__ dt,
    const float* __restrict__ trap,
    const float* __restrict__ q_bias,
    const float* __restrict__ k_bias,
    const float* __restrict__ angles,
    const float* __restrict__ mimo_v,
    const float* __restrict__ d,
    const float* __restrict__ z,
    const float* __restrict__ angle_state,
    const float* __restrict__ ssm_state,
    const float* __restrict__ k_state,
    const float* __restrict__ v_state,
    const uint32_t batch,
    const uint32_t rank,
    const uint32_t nheads,
    const uint32_t headdim,
    const uint32_t d_state,
    const uint32_t headdim_angles,
    const uint32_t has_d,
    const uint32_t has_z
) {
    const uint32_t head = blockIdx.x;
    const uint32_t b = blockIdx.y;
    if (head >= nheads || b >= batch) return;

    const uint32_t base = b * nheads + head;
    float dt_val = dt[base];
    float angles_local[64];
    for (uint32_t i = 0; i < headdim_angles; ++i) {
        float a = angle_state[base * headdim_angles + i];
        a += mamba3_tanh(angles[base * headdim_angles + i]) * 3.141592653589793f * dt_val;
        angles_local[i] = a;
        out_angle_state[base * headdim_angles + i] = a;
    }

    float adt_val = adt[base];
    float trap_val = mamba3_sigmoid(trap[base]);
    float alpha = expf(adt_val);
    float beta = alpha * dt_val * (1.f - trap_val);
    float gamma = trap_val * dt_val;

    const uint32_t state_base = base * headdim * d_state;

    for (uint32_t vi = 0; vi < headdim; ++vi) {
        float v_prev = v_state[base * headdim + vi];
        float v_cur = v[base * headdim + vi];
        for (uint32_t ki = 0; ki < d_state; ++ki) {
            float diff = 0.f;
            for (uint32_t r = 0; r < rank; ++r) {
                uint32_t qk_base = ((b * rank + r) * nheads + head) * d_state + ki;
                float k_prev = k_state[qk_base];
                float k_cur = k[qk_base] + k_bias[(r * nheads + head) * d_state + ki];
                float v_r = v_cur * mimo_v[(head * rank + r) * headdim + vi];
                diff += beta * v_prev * k_prev + gamma * v_r * k_cur;
                out_k_state[qk_base] = k_cur;
            }
            uint32_t idx = state_base + vi * d_state + ki;
            out_ssm_state[idx] = ssm_state[idx] * alpha + diff;
        }
    }

    for (uint32_t vi = 0; vi < headdim; ++vi) {
        float acc = 0.f;
        for (uint32_t r = 0; r < rank; ++r) {
            for (uint32_t ki = 0; ki < d_state; ++ki) {
                uint32_t qk_base = ((b * rank + r) * nheads + head) * d_state + ki;
                acc += out_ssm_state[state_base + vi * d_state + ki]
                    * (q[qk_base] + q_bias[(r * nheads + head) * d_state + ki]);
            }
        }
        if (has_d) acc += d[head] * v[base * headdim + vi];
        if (has_z) acc *= mamba3_silu(z[base * headdim + vi]);
        out[base * headdim + vi] = acc;
    }
}
