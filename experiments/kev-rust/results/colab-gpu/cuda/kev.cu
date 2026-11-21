// FP32 inference kernels. Compile without FMA contraction to follow the CPU
// recurrence's multiplication/addition rounding closely.
extern "C" __global__ void kev_depthwise(
    const float *x, const float *w, float *out,
    unsigned int count, unsigned int channels, unsigned int time,
    unsigned int width, unsigned int len) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    unsigned int bc = index / len;
    unsigned int t = index % len;
    float sum = 0.0f;
    for (unsigned int j = 0; j < width; ++j)
        sum += x[bc * time + t + j] * w[(bc % channels) * width + j];
    out[index] = sum;
}

// Each warp owns a tile of 32 value columns from one (batch, head) state.
// Columns are independent: no atomics or inter-thread synchronization is needed.
// Input packs q, k, v, decay, beta, initial state. Output packs readouts and
// final state. The initial state is immutable, preserving prefix cache forks.
extern "C" __global__ void kev_delta_scan(
    const float *input, float *output, unsigned int bh,
    unsigned int len, unsigned int kd, unsigned int vd) {
    const unsigned int head = blockIdx.x;
    const unsigned int column = blockIdx.y * 32 + threadIdx.x;
    if (column >= vd) return;
    const unsigned int qn = bh * len * kd;
    const unsigned int vn = bh * len * vd;
    const unsigned int gn = bh * len;
    const float *q = input;
    const float *k = q + qn;
    const float *v = k + qn;
    const float *g = v + vn;
    const float *beta = g + gn;
    const float *initial = beta + gn;
    float *final_state = output + vn;
    extern __shared__ float state[];
    for (unsigned int i = 0; i < kd; ++i)
        state[i * 32 + threadIdx.x] = initial[(head * kd + i) * vd + column];
    for (unsigned int t = 0; t < len; ++t) {
        const unsigned int token = head * len + t;
        float memory = 0.0f;
        for (unsigned int i = 0; i < kd; ++i) {
            float &s = state[i * 32 + threadIdx.x];
            s *= g[token];
            memory += s * k[token * kd + i];
        }
        const float delta = (v[token * vd + column] - memory) * beta[token];
        float readout = 0.0f;
        for (unsigned int i = 0; i < kd; ++i) {
            float &s = state[i * 32 + threadIdx.x];
            s += k[token * kd + i] * delta;
            readout += s * q[token * kd + i];
        }
        output[token * vd + column] = readout;
    }
    for (unsigned int i = 0; i < kd; ++i)
        final_state[(head * kd + i) * vd + column] = state[i * 32 + threadIdx.x];
}
