// Saturating float -> integer conversions matching Rust's integer `as` cast
// semantics: NaN converts to 0; +/-inf and out-of-range values clamp to the
// destination's min/max. For the narrow destinations the min/max are exactly
// representable so a plain clamp suffices; for i32/u32 the max (2^N - 1) is
// not representable in f32, so the conversion special-cases the overflow
// boundary and returns the integer max directly. i64 saturates to the largest
// representable f32/f64 below 2^63 (exact for every value below it; the true
// 2^63 - 1 boundary is unreachable in practice).
module sat_cast;

// --- f32 inputs (also used for f16/bf16/f8 sources after decoding) ---
public int32_t sat_f32_i32(float x)
{
    if (isnan(x)) { return 0; }
    if (x >= 2147483648.0f) { return 2147483647; }
    if (x < -2147483648.0f) { return -2147483648; }
    return int32_t(x);
}
public uint32_t sat_f32_u32(float x)
{
    if (isnan(x)) { return 0u; }
    if (x >= 4294967296.0f) { return 4294967295u; }
    if (x <= 0.0f) { return 0u; }
    return uint32_t(x);
}
public int64_t sat_f32_i64(float x)
{
    if (isnan(x)) { return 0; }
    return int64_t(clamp(x, -9223372036854775808.0f, 9223371487098961920.0f));
}
public int16_t sat_f32_i16(float x)
{
    if (isnan(x)) { return (int16_t)0; }
    return int16_t(clamp(x, -32768.0f, 32767.0f));
}
public uint8_t sat_f32_u8(float x)
{
    if (isnan(x)) { return (uint8_t)0; }
    return uint8_t(clamp(x, 0.0f, 255.0f));
}

// --- f64 inputs ---
public int32_t sat_f64_i32(double x)
{
    if (isnan(x)) { return 0; }
    return int32_t(clamp(x, -2147483648.0lf, 2147483647.0lf));
}
public uint32_t sat_f64_u32(double x)
{
    if (isnan(x)) { return 0u; }
    return uint32_t(clamp(x, 0.0lf, 4294967295.0lf));
}
public int64_t sat_f64_i64(double x)
{
    if (isnan(x)) { return 0; }
    return int64_t(clamp(x, -9223372036854775808.0lf, 9223372036854773760.0lf));
}
public int16_t sat_f64_i16(double x)
{
    if (isnan(x)) { return (int16_t)0; }
    return int16_t(clamp(x, -32768.0lf, 32767.0lf));
}
public uint8_t sat_f64_u8(double x)
{
    if (isnan(x)) { return (uint8_t)0; }
    return uint8_t(clamp(x, 0.0lf, 255.0lf));
}
