#include <vulkano.slang>

struct VertexOutput {
    let tex_coords: float2;
}

static const let ACES_INPUT_MATRIX: float3x3 = {
    { 0.59719, 0.35458, 0.04823 },
    { 0.07600, 0.90834, 0.01566 },
    { 0.02840, 0.13383, 0.83777 },
};

static const let ACES_OUTPUT_MATRIX: float3x3 = {
    {  1.60475, -0.53108, -0.07367 },
    { -0.10208,  1.10813, -0.00605 },
    { -0.00327, -0.07276,  1.07602 },
};

struct PushConstants {
    let sampler_id: SamplerId;
    let texture_id: SampledImageId;
    let exposure: float;
}

[shader("fragment")]
func main(uniform params: PushConstants, input: VertexOutput) -> float4 {
    let texture = vko::uniformTexture2D<float4>(params.texture_id);
    let sampler_state = vko::uniformSamplerState(params.sampler_id);
    let hdr_color = params.exposure * texture.Sample(sampler_state, input.tex_coords);

    var color = mul(ACES_INPUT_MATRIX, hdr_color.rgb);
    color = rrtAndOdtFit(color);
    color = mul(ACES_OUTPUT_MATRIX, color);

    return float4(color, 1.0);
}

func rrtAndOdtFit(v: float3) -> float3 {
    let a = v * (v + 0.0245786) - 0.000090537;
    let b = v * (0.983729 * v + 0.4329510) + 0.238081;
    return a / b;
}
