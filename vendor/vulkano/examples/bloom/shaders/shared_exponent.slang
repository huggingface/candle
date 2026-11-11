// https://registry.khronos.org/OpenGL/extensions/EXT/EXT_texture_shared_exponent.txt
func convertToSharedExponent(color: float3) -> uint {
    static const let MAX: int = 65408;
    static const let BIAS: int = 15;
    static const let MANTISSA_BITS: int = 9;
    static const let MANTISSA_VALUES: int = 512;

    let clamped_color = clamp(color, float3(0.0), float3(MAX));
    let max_clamped_component = max(clamped_color.r, max(clamped_color.g, clamped_color.b));
    let max_clamped_exponent = int((bit_cast<uint>(max_clamped_component) >> 23) & 0xFF) - 127;
    var shared_exponent = max(-BIAS - 1, max_clamped_exponent) + 1 + BIAS;
    var divisor = exp2(float(shared_exponent - BIAS - MANTISSA_BITS));
    let max_shared_component = int(floor(max_clamped_component / divisor + 0.5));

    if (max_shared_component == MANTISSA_VALUES) {
        shared_exponent += 1;
        divisor *= 2.0;
    }

    let shared_color = floor(clamped_color / divisor + 0.5);

    return (uint(shared_exponent) << 27)
        | (uint(shared_color.b) << 18)
        | (uint(shared_color.g) << 9)
        | (uint(shared_color.r) << 0);
}

