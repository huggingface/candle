#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout (binding = 1) uniform sampler2D samplerColor;

layout (binding = 0) uniform UBO{
    vec3 color;
} ubo;


layout (location = 0) in vec2 o_uv;
layout (location = 0) out vec4 uFragColor;

void main() {
    vec4 color = texture(samplerColor, o_uv);
    uFragColor = color;
}
