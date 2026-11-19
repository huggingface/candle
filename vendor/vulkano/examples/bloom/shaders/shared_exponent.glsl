// https://registry.khronos.org/OpenGL/extensions/EXT/EXT_texture_shared_exponent.txt
uint convertToSharedExponent(vec3 color) {
    const int MAX = 65408;
    const int BIAS = 15;
    const int MANTISSA_BITS = 9;
    const int MANTISSA_VALUES = 512;

    vec3 clamped_color = clamp(color, vec3(0.0), vec3(MAX));
    float max_clamped_component = max(clamped_color.r, max(clamped_color.g, clamped_color.b));
    int max_clamped_exponent = int((floatBitsToUint(max_clamped_component) >> 23) & 0xFF) - 127;
    int shared_exponent = max(-BIAS - 1, max_clamped_exponent) + 1 + BIAS;
    float divisor = exp2(float(shared_exponent - BIAS - MANTISSA_BITS));
    int max_shared_component = int(floor(max_clamped_component / divisor + 0.5));

    if (max_shared_component == MANTISSA_VALUES) {
        shared_exponent += 1;
        divisor *= 2.0;
    }

    vec3 shared_color = floor(clamped_color / divisor + 0.5);

    return (uint(shared_exponent) << 27)
        | (uint(shared_color.b) << 18)
        | (uint(shared_color.g) << 9)
        | (uint(shared_color.r) << 0);
}
