#version 450
#include <vulkano.glsl>

layout(location = 0) in vec2 v_tex_coords;

layout(location = 0) out vec4 f_color;

layout(push_constant) uniform PushConstants {
    SamplerId sampler_id;
    SampledImageId texture_id;
    float exposure;
};

const mat3 ACES_INPUT_MATRIX = {
    { 0.59719, 0.07600, 0.02840 },
    { 0.35458, 0.90834, 0.13383 },
    { 0.04823, 0.01566, 0.83777 },
};

const mat3 ACES_OUTPUT_MATRIX = {
    {  1.60475, -0.10208, -0.00327 },
    { -0.53108,  1.10813, -0.07276 },
    { -0.07367, -0.00605,  1.07602 },
};

vec3 rrtAndOdtFit(vec3 v) {
    vec3 a = v * (v + 0.0245786) - 0.000090537;
    vec3 b = v * (0.983729 * v + 0.4329510) + 0.238081;
    return a / b;
}

void main() {
    vec4 hdr_color = exposure * texture(vko_sampler2D(texture_id, sampler_id), v_tex_coords);

    vec3 color = ACES_INPUT_MATRIX * hdr_color.rgb;
    color = rrtAndOdtFit(color);
    color = ACES_OUTPUT_MATRIX * color;

    f_color = vec4(color, 1.0);
}
