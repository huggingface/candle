#include <metal_stdlib>
using namespace metal;

// Constants
// 2^32 and 1/2^32. Useful for converting between float and uint.
static constexpr constant ulong UNIF01_NORM32 = 4294967296;
static constexpr constant float UNIF01_INV32 = 2.328306436538696289e-10;
// 2 * pi
static constexpr constant float TWO_PI = 2.0 * M_PI_F;
static constexpr constant int3 S1 = {13, 19, 12};
static constexpr constant int3 S2 = {2, 25, 4};
static constexpr constant int3 S3 = {3, 11, 17};

static constexpr constant uint64_t PHI[16] = {
    0x9E3779B97F4A7C15,
    0xF39CC0605CEDC834,
    0x1082276BF3A27251,
    0xF86C6A11D0C18E95,
    0x2767F0B153D27B7F,
    0x0347045B5BF1827F,
    0x01886F0928403002,
    0xC1D64BA40F335E36,
    0xF06AD7AE9717877E,
    0x85839D6EFFBD7DC6,
    0x64D325D1C5371682,
    0xCADD0CCCFDFFBBE1,
    0x626E33B8D04B4331,
    0xBBF73C790D94F79D,
    0x471C4AB3ED3D82A5,
    0xFEC507705E4AE6E5,
};

// Combined Tausworthe and LCG Random Number Generator.
// https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-37-efficient-random-number-generation-and-application
// https://indico.cern.ch/event/93877/contributions/2118070/attachments/1104200/1575343/acat3_revised_final.pdf
class HybridTaus {
private:
    thread float seed;

    // Generate seeds for each thread.
    thread uint4 seed_per_thread(const ulong4 seeds) {
        return uint4(ulong4(seeds) * ulong4(PHI[0], PHI[1], PHI[2], PHI[3]) * ulong4(1099087573UL));
    }

    // Tausworthe generator.
    thread uint taus(const uint z, const int3 s, const uint M) {
        uint b = (((z << s.x) ^ z) >> s.y);
        return (((z & M) << s.z) ^ b);
    }

    // LCG generator.
    thread uint lcg(const uint z) {
        return (1664525 * z + 1013904223UL);
    }

public:
    thread HybridTaus(const ulong4 seeds) {
        uint4 seed = this->seed_per_thread(seeds);

        // Seed #1
        uint z1 = taus(seed.x, S1, 4294967294UL);
        uint z2 = taus(seed.y, S2, 4294967288UL);
        uint z3 = taus(seed.z, S3, 4294967280UL);
        uint z4 = lcg(seed.x);

        // Seed #2
        uint r1 = (z1^z2^z3^z4^seed.y);
        z1 = taus(r1, S1, 429496729UL);
        z2 = taus(r1, S2, 4294967288UL);
        z3 = taus(r1, S3, 429496280UL);
        z4 = lcg(r1);

        // Seed #3
        r1 = (z1^z2^z3^z4^seed.z);
        z1 = taus(r1, S1, 429496729UL);
        z2 = taus(r1, S2, 4294967288UL);
        z3 = taus(r1, S3, 429496280UL);
        z4 = lcg(r1);

        // Seed #4
        r1 = (z1^z2^z3^z4^seed.w);
        z1 = taus(r1, S1, 429496729UL);
        z2 = taus(r1, S2, 4294967288UL);
        z3 = taus(r1, S3, 429496280UL);
        z4 = lcg(r1);

        this->seed = (z1^z2^z3^z4) * UNIF01_INV32;
    }

    thread float rand() {
        uint seed = this->seed * UNIF01_NORM32;
        uint z1 = taus(seed, S1, 429496729UL);
        uint z2 = taus(seed, S2, 4294967288UL);
        uint z3 = taus(seed, S3, 429496280UL);
        uint z4 = lcg(seed);

        thread float old_seed = this->seed;
        this->seed = (z1^z2^z3^z4) * UNIF01_INV32;
        return old_seed;
    }
};

template<typename T> METAL_FUNC void rand_uniform(
    constant size_t &elem_count,
    constant ulong &seed,
    constant float &min,
    constant float &max,
    device T *out,
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= elem_count) {
        return;
    }
    float diff = max - min;
    HybridTaus rng = HybridTaus({seed, tid, 1, 1});
    out[tid] = static_cast<T>(rng.rand() * diff + min);
}

#define UNIFORM_OP(NAME, T)                                 \
kernel void rand_uniform_##NAME(                            \
    constant size_t &elem_count,                            \
    constant ulong &seed,                                   \
    constant float &min,                                    \
    constant float &max,                                    \
    device T *out,                                          \
    uint tid [[thread_position_in_grid]]                    \
) {                                                         \
    rand_uniform<T>(elem_count, seed, min, max, out, tid);  \
}                                                           \

#define RANDOM_OPS(NAME, T) \
UNIFORM_OP(NAME, T)         \

RANDOM_OPS(f32, float)
RANDOM_OPS(f16, half)

#if __METAL_VERSION__ >= 310
RANDOM_OPS(bf16, bfloat)
#endif
