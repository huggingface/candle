// -*- Metal -*-
//===-- metal_simdgroup_event ---------------------------------------------===//
// Copyright (c) 2024 Philip Turner. See MIT LICENSE
//===----------------------------------------------------------------------===//

#ifndef __METAL_SIMDGROUP_EVENT
#define __METAL_SIMDGROUP_EVENT

// Invoking the generation of LLVM bitcode for async copies.
//
//   %struct._simdgroup_event_t = type opaque
//
struct _simdgroup_event_t;

// Invoking the generation of LLVM bitcode for async copies.
//
//   Bitcode: TBD
//
thread _simdgroup_event_t*
__metal_simdgroup_async_copy_1d(
  ulong, ulong, threadgroup void *, const device void *, ulong)
  __asm("air.simdgroup_async_copy_1d.p3i8.p1i8");

// Invoking the generation of LLVM bitcode for async copies.
//
//   Bitcode: TBD
//
thread _simdgroup_event_t*
__metal_simdgroup_async_copy_1d(
  ulong, ulong, device void *, const threadgroup void *, ulong)
  __asm("air.simdgroup_async_copy_1d.p1i8.p3i8");

// Invoking the generation of LLVM bitcode for async copies.
//
//   ; Function Attrs: argmemonly convergent nounwind
//   declare %struct._simdgroup_event_t*
//     @air.simdgroup_async_copy_2d.p3i8.p1i8(
//       i64, i64,
//       i8 addrspace(3)* nocapture writeonly, i64, i64, <2 x i64>,
//       i8 addrspace(1)* nocapture readonly,  i64, i64, <2 x i64>,
//       <2 x i64>, i32)
//     local_unnamed_addr #4
//
thread _simdgroup_event_t*
__metal_simdgroup_async_copy_2d(
  ulong, ulong,
  threadgroup void *, ulong, ulong, ulong2,
  const device void *, ulong, ulong, ulong2,
  long2, int)
  __asm("air.simdgroup_async_copy_2d.p3i8.p1i8");

// Invoking the generation of LLVM bitcode for async copies.
//
//   ; Function Attrs: argmemonly convergent nounwind
//   declare %struct._simdgroup_event_t*
//     @air.simdgroup_async_copy_2d.p1i8.p3i8(
//       i64, i64,
//       i8 addrspace(1)* nocapture writeonly, i64, i64, <2 x i64>,
//       i8 addrspace(3)* nocapture readonly,  i64, i64, <2 x i64>,
//       <2 x i64>, i32)
//     local_unnamed_addr #4
//
thread _simdgroup_event_t*
__metal_simdgroup_async_copy_2d(
  ulong, ulong,
  device void *, ulong, ulong, ulong2,
  const threadgroup void *, ulong, ulong, ulong2,
  long2, int)
  __asm("air.simdgroup_async_copy_2d.p1i8.p3i8");

// Invoking the generation of LLVM bitcode for async copies.
//
//   ; Function Attrs: convergent nounwind
//   declare void
//     @air.wait_simdgroup_events(i32, %struct._simdgroup_event_t** nocapture)
//     local_unnamed_addr #3
//
void __metal_wait_simdgroup_events(
  int, thread _simdgroup_event_t**)
  __asm("air.wait_simdgroup_events");

#pragma METAL internals : enable
namespace metal
{
  enum class simdgroup_async_copy_clamp_mode {
    clamp_to_zero = 0,
    clamp_to_edge = 1
  };

  struct simdgroup_event {
    METAL_FUNC simdgroup_event() thread {}

    template <typename T>
    METAL_FUNC void async_copy(
      threadgroup T *dst,
      const device T *src,
      ulong n_elements
    ) thread {
      event = __metal_simdgroup_async_copy_1d(
        // Description of the data type.
        sizeof(T),
        alignof(T),

        // Description of the arguments.
        reinterpret_cast<threadgroup void *>(dst),
        reinterpret_cast<const device void *>(src),
        n_elements);
    }

    template <typename T>
    METAL_FUNC void async_copy(
      device T *dst,
      const threadgroup T *src,
      ulong n_elements
    ) thread {
      event = __metal_simdgroup_async_copy_1d(
        // Description of the data type.
        sizeof(T),
        alignof(T),

        // Description of the arguments.
        reinterpret_cast<device void *>(dst),
        reinterpret_cast<const threadgroup void *>(src),
        n_elements);
    }

    template <typename T>
    METAL_FUNC void async_copy(
      // Description of the destination.
      threadgroup T *dst,
      ushort dst_elements_per_row,
      ushort2 dst_tile_dimensions,

      // Description of the source.
      const device T *src,
      uint src_elements_per_row,
      ushort2 src_tile_dimensions,

      // Other arguments.
      bool transpose_matrix = false,
      simdgroup_async_copy_clamp_mode clamp_mode =
        simdgroup_async_copy_clamp_mode::clamp_to_zero
    ) thread {
      if (transpose_matrix) {
        src_tile_dimensions = src_tile_dimensions.yx;
        dst_tile_dimensions = dst_tile_dimensions.yx;
      }
      event = __metal_simdgroup_async_copy_2d(
        // Description of the data type.
        sizeof(T),
        alignof(T),

        // Description of the destination.
        reinterpret_cast<threadgroup void *>(dst),
        ushort(dst_elements_per_row),
        1,
        ulong2(dst_tile_dimensions),

        // Description of the source.
        reinterpret_cast<const device void *>(src),
        uint(src_elements_per_row),
        1,
        ulong2(src_tile_dimensions),

        // Other arguments.
        long2(0),
        static_cast<int>(clamp_mode));
    }

    template <typename T>
    METAL_FUNC void async_copy(
      // Description of the destination.
      device T *dst,
      uint dst_elements_per_row,
      ushort2 dst_tile_dimensions,

      // Description of the source.
      const threadgroup T *src,
      ushort src_elements_per_row,
      ushort2 src_tile_dimensions,

      // Other arguments.
      bool transpose_matrix = false
    ) thread {
      if (transpose_matrix) {
        src_tile_dimensions = src_tile_dimensions.yx;
        dst_tile_dimensions = dst_tile_dimensions.yx;
      }
      event = __metal_simdgroup_async_copy_2d(
        // Description of the data type.
        sizeof(T),
        alignof(T),

        // Description of the destination.
        reinterpret_cast<device void *>(dst),
        uint(dst_elements_per_row),
        1,
        ulong2(dst_tile_dimensions),

        // Description of the source.
        reinterpret_cast<const threadgroup void *>(src),
        ushort(src_elements_per_row),
        1,
        ulong2(src_tile_dimensions),

        // Other arguments.
        long2(0),
        0);
    }

    METAL_FUNC static void wait(int count, thread simdgroup_event *events) {
      __metal_wait_simdgroup_events(
        count, reinterpret_cast<thread _simdgroup_event_t**>(events));
    }

  private:
    // Invoking the generation of LLVM bitcode for async copies.
    //
    //   %"struct.metal::simdgroup_event" = type { %struct._simdgroup_event_t* }
    //
    thread _simdgroup_event_t* event;
  };
} // namespace metal
#pragma METAL internals : disable

#endif
