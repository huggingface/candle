#include <vulkano.slang>
#include "shared_exponent.slang"

static const let EPSILON: float = 1.19209290e-07;

[shader("compute")]
[numthreads(8, 8, 1)]
func downsample(dst_coord: uint2 : SV_DispatchThreadId, uniform params: DownsampleParams) {
    let dst_texture = vko::uniformWTexture2D<uint>(params.dst_mip_image_id);
    var dst_size: uint2;
    dst_texture.GetDimensions(dst_size.x, dst_size.y);

    if (dst_coord.x >= dst_size.x || dst_coord.y >= dst_size.y) {
        return;
    }

    let src_texture = vko::uniformTexture2D(params.texture_id);
    var src_size: uint2;
    var _: uint;
    src_texture.GetDimensions(params.srcMipLevel(), src_size.x, src_size.y, _);
    let src_texel_size = 1.0 / float2(src_size);
    let uv = (float2(dst_coord) + 0.5) / float2(dst_size);
    var color = params.downsampleBox13(uv, src_texel_size);

    if (params.srcMipLevel() == 0) {
        color = params.prefilter(color);
    }

    color = max(color, 0.0001);

    params.store(dst_coord, color);
}

struct DownsampleParams {
    let sampler_id: SamplerId;
    let texture_id: SampledImageId;
    let dst_mip_image_id: StorageImageId;
    let dst_mip_level: uint;
    let threshold: float;
    let knee: float;

    // 13-tap box filter with the sample locations and corresponding weights depicted below. This
    // is Unity's preferred downsample bloom filter. According to them, it has "better, temporally
    // stable box filtering." It's not even a box filter, but it sure looks better than one. A lot
    // of people like the way Unity's bloom looks, so that's the example we're going with.
    //
    // ┌───┬───┬───┬───┬───┐
    // │ 1 │   │ 2 │   │ 1 │
    // ├───┼───┼───┼───┼───┤
    // │   │ 4 │   │ 4 │   │
    // ├───┼───┼───┼───┼───┤
    // │ 2 │   │ 4 │   │ 2 │
    // ├───┼───┼───┼───┼───┤
    // │   │ 4 │   │ 4 │   │
    // ├───┼───┼───┼───┼───┤
    // │ 1 │   │ 2 │   │ 1 │
    // └───┴───┴───┴───┴───┘
    func downsampleBox13(uv: float2, src_texel_size: float2) -> float3 {
        var color: float3;
        color  = this.sample1(uv + float2(-2.0, -2.0) * src_texel_size) * 1.0;
        color += this.sample1(uv + float2( 0.0, -2.0) * src_texel_size) * 2.0;
        color += this.sample1(uv + float2( 2.0, -2.0) * src_texel_size) * 1.0;
        color += this.sample1(uv + float2(-1.0, -1.0) * src_texel_size) * 4.0;
        color += this.sample1(uv + float2( 1.0, -1.0) * src_texel_size) * 4.0;
        color += this.sample1(uv + float2(-2.0,  0.0) * src_texel_size) * 2.0;
        color += this.sample1(uv + float2( 0.0,  0.0) * src_texel_size) * 4.0;
        color += this.sample1(uv + float2( 2.0,  0.0) * src_texel_size) * 2.0;
        color += this.sample1(uv + float2(-1.0,  1.0) * src_texel_size) * 4.0;
        color += this.sample1(uv + float2( 1.0,  1.0) * src_texel_size) * 4.0;
        color += this.sample1(uv + float2(-2.0,  2.0) * src_texel_size) * 1.0;
        color += this.sample1(uv + float2( 0.0,  2.0) * src_texel_size) * 2.0;
        color += this.sample1(uv + float2( 2.0,  2.0) * src_texel_size) * 1.0;

        return color / 32.0;
    }

    func sample1(uv: float2) -> float3 {
        let texture = vko::uniformTexture2D(this.texture_id);
        let sampler_state = vko::uniformSamplerState(this.sampler_id);
        return texture.SampleLevel(sampler_state, uv, this.srcMipLevel()).rgb;
    }

    // This is Unity's prefilter of choice, except without the auto-exposure that you would want to
    // do in a real application. It's not really a *pre*filter though as it runs on the downsampled
    // color.
    func prefilter(color: float3) -> float3 {
        return this.quadraticThreshold(color);
    }

    func quadraticThreshold(float3 color) -> float3 {
        let brightness = max(color.r, max(color.g, color.b));
        var quadratic_response =
            clamp(brightness - this.threshold + this.knee, 0.0, 2.0 * this.knee);
        quadratic_response = (quadratic_response * quadratic_response) / (0.25 / this.knee);

        color *= max(quadratic_response, brightness - this.threshold) / max(brightness, EPSILON);

        return color;
    }

    func store(dst_coord: uint2, color: float3) {
        let texture = vko::uniformWTexture2D<uint>(this.dst_mip_image_id);
        let packed = convertToSharedExponent(color);
        texture.Store(dst_coord, packed);
    }

    func srcMipLevel() -> uint {
        return this.dst_mip_level - 1;
    }
}

[shader("compute")]
[numthreads(8, 8, 1)]
func upsample(dst_coord: uint2 : SV_DispatchThreadID, uniform params: UpsampleParams) {
    let dst_texture = vko::uniformWTexture2D<uint>(params.dst_mip_image_id);
    var dst_size: uint2;
    dst_texture.GetDimensions(dst_size.x, dst_size.y);

    if (dst_coord.x >= dst_size.x || dst_coord.y >= dst_size.y) {
        return;
    }

    let src_texture = vko::uniformTexture2D(params.texture_id);
    var src_size: uint2;
    var _: uint;
    src_texture.GetDimensions(params.dst_mip_level, src_size.x, src_size.y, _);
    let src_texel_size = 1.0 / float2(src_size);
    let uv = (float2(dst_coord) + 0.5) / float2(dst_size);
    var color = params.upsampleTent9(uv, src_texel_size);

    color *= params.intensity;

    params.blend(uv, dst_coord, color);
}

struct UpsampleParams {
    let sampler_id: SamplerId;
    let texture_id: SampledImageId;
    let dst_mip_image_id: StorageImageId;
    let dst_mip_level: uint;
    let intensity: float;

    // 9-tap tent filter with the locations and corresponding weights depicted below. This is
    // Unity's preferred upsample bloom filter. This filter is really nice for upsampling because
    // it can be implemented with just 4 taps with identical results (although Unity didn't do this
    // optimization for some reason). And keep in mind that the upsample filter's performance is
    // more important than the downsample filter's performance as the former writes to the
    // highest-resolution image unlike the latter. This cheap but well-working filter paired with
    // the much more expensive 13-tap filter on the way down results in a very good outcome.
    //
    // ┌───┬───┬───┐
    // │ 1 │ 2 │ 1 │
    // ├───┼───┼───┤
    // │ 2 │ 4 │ 2 │
    // ├───┼───┼───┤
    // │ 1 │ 2 │ 1 │
    // └───┴───┴───┘
    func upsampleTent9(uv: float2, src_texel_size: float2) -> float3 {
        var color: float3;
        color  = this.sample1(uv + float2(-0.5, -0.5) * src_texel_size);
        color += this.sample1(uv + float2( 0.5, -0.5) * src_texel_size);
        color += this.sample1(uv + float2(-0.5,  0.5) * src_texel_size);
        color += this.sample1(uv + float2( 0.5,  0.5) * src_texel_size);

        return color / 4.0;
    }

    func sample1(uv: float2) -> float3 {
        let texture = vko::uniformTexture2D(this.texture_id);
        let sampler_state = vko::uniformSamplerState(this.sampler_id);
        return texture.SampleLevel(sampler_state, uv, this.srcMipLevel()).rgb;
    }

    func blend(uv: float2, dst_coord: uint2, color: float3) {
        let texture = vko::uniformTexture2D(this.texture_id);
        let sampler_state = vko::uniformSamplerState(this.sampler_id);
        let old_color = texture.SampleLevel(sampler_state, uv, this.dst_mip_level).rgb;
        this.store(dst_coord, old_color + color);
    }

    func store(dst_coord: uint2, color: float3) {
        let texture = vko::uniformWTexture2D<uint>(this.dst_mip_image_id);
        let packed = convertToSharedExponent(color);
        texture.Store(dst_coord, packed);
    }

    func srcMipLevel() -> uint {
        return this.dst_mip_level + 1;
    }
}
