#version 460
#extension GL_EXT_ray_tracing : require
#define VKO_ACCELERATION_STRUCTURE_ENABLED 1
#include <vulkano.glsl>

layout(location = 0) rayPayloadEXT vec3 hit_value;

VKO_DECLARE_STORAGE_BUFFER(camera, Camera {
    // Camera view * projection
    mat4 view_proj;
    // Camera inverse view matrix
    mat4 view_inverse;
    // Camera inverse projection matrix
    mat4 proj_inverse;
})

layout(push_constant) uniform PushConstants {
    StorageImageId image_id;
    AccelerationStructureId acceleration_structure_id;
    StorageBufferId camera_buffer_id;
};

#define camera vko_buffer(camera, camera_buffer_id)

void main() {
    const vec2 pixel_center = vec2(gl_LaunchIDEXT.xy) + vec2(0.5);
    const vec2 in_uv = pixel_center / vec2(gl_LaunchSizeEXT.xy);
    vec2 d = in_uv * 2.0 - 1.0;

    vec4 origin = camera.view_inverse * vec4(0, 0, 0, 1);
    vec4 target = camera.proj_inverse * vec4(d.x, d.y, 1, 1);
    vec4 direction = camera.view_inverse * vec4(normalize(target.xyz), 0);

    uint ray_flags = gl_RayFlagsOpaqueEXT;
    float t_min = 0.001;
    float t_max = 10000.0;

    traceRayEXT(
        // acceleration structure
        vko_accelerationStructureEXT(acceleration_structure_id),
        // rayFlags
        ray_flags,
        // cullMask
        0xFF,
        // sbtRecordOffset
        0,
        // sbtRecordStride
        0,
        // missIndex
        0,
        // ray origin
        origin.xyz,
        // ray min range
        t_min,
        // ray direction
        direction.xyz,
        // ray max range
        t_max,
        // payload (location = 0)
        0);

    imageStore(vko_image2D_rgba32f(image_id), ivec2(gl_LaunchIDEXT.xy), vec4(hit_value, 1.0));
}
