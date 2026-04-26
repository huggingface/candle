#include <metal_stdlib>
using namespace metal;

template<uint Y>
constexpr uint div_ceil(uint x) {
    return x / Y + (x % Y > 0);
}

template<uint X, uint Y>
constexpr uint div_ceil() {
    return X / Y + (X % Y > 0);
}

template<typename T>
constexpr uint work_per_thread() {
    return div_ceil<8, sizeof(T)>();
}

// Each thread handles `W` vocab positions and scans the deduped context list.
// If the position appears in context, the repetition penalty is
// applied in the same manner as the original cpu impl:
//   logit >  0 -> logit / penalty
//   logit <= 0 -> logit * penalty
template <typename T, uint W = work_per_thread<T>()>
[[kernel]] void repeat_penalty(
    device const T *input [[buffer(0)]],
    device T *output [[buffer(1)]],
    device const uint *context [[buffer(2)]],
    constant uint &vocab_size [[buffer(3)]],
    constant uint &context_size [[buffer(4)]],
    constant float &penalty [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    const uint step = div_ceil<W>(vocab_size);
    #pragma clang loop unroll(full)
    for (uint i = tid; i < vocab_size; i += step) {
        float logit = float(input[i]);

        // Attempt to base `context` read starting point on `tid` and reduce conflicts. Probably unsuccessfully.
        uint c = i + 1;
        uint j = 0;
        while (c != i) {
            if (c >= context_size) {
                c = 0;
            }
            if (context[j] == i) {
                logit = (logit > 0.0f) ? (logit / penalty) : (logit * penalty);
                break;
            }
            c++;
        }
        output[i] = T(logit);
    }
}

#define init_repeat_penalty(tname, t) \
    template [[host_name("repeat_penalty_" #tname)]] [[kernel]] decltype(repeat_penalty<t>) repeat_penalty<t>;

init_repeat_penalty(f32, float);
init_repeat_penalty(f16, half);

#if defined(__HAVE_BFLOAT__)
init_repeat_penalty(bf16, bfloat);
#endif
