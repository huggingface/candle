// -*- Metal -*-
//===-- metal_simdgroup_matrix_storage ------------------------------------===//
// Copyright (c) 2024 Philip Turner. See MIT LICENSE
//===----------------------------------------------------------------------===//

#ifndef __METAL_SIMDGROUP_MATRIX_STORAGE
#define __METAL_SIMDGROUP_MATRIX_STORAGE

#pragma METAL internals : enable
namespace metal
{
  template <typename T>
  struct simdgroup_matrix_storage {
    typedef vec<T, 64> storage_type;

    storage_type t;

    METAL_FUNC thread vec<T, 2>* thread_elements() thread {
      return reinterpret_cast<thread vec<T, 2>*>(&t);
    }

    METAL_FUNC simdgroup_matrix_storage() thread = default;

    METAL_FUNC simdgroup_matrix_storage(vec<T, 2> thread_elements) thread {
      *(this->thread_elements()) = thread_elements;
    }

    METAL_FUNC static device T* apply_offset(device T *src, uint elements_per_row, uint2 matrix_origin, bool transpose_matrix = false) {
      if (transpose_matrix) {
        return src + ulong(matrix_origin.x * elements_per_row) + matrix_origin.y;
      } else {
        return src + ulong(matrix_origin.y * elements_per_row) + matrix_origin.x;
      }
    }

    METAL_FUNC static threadgroup T* apply_offset(threadgroup T *src, ushort elements_per_row, ushort2 matrix_origin, bool transpose_matrix = false) {
      if (transpose_matrix) {
        return src + matrix_origin.x * elements_per_row + matrix_origin.y;
      } else {
        return src + matrix_origin.y * elements_per_row + matrix_origin.x;
      }
    }

    template <typename U>
    METAL_FUNC void load(const device U *src, uint elements_per_row, ushort2 matrix_origin, bool transpose_matrix = false) {
      if (transpose_matrix) {
        uint address0 = uint(matrix_origin.x + 0) * elements_per_row + uint(matrix_origin.y);
        uint address1 = uint(matrix_origin.x + 1) * elements_per_row + uint(matrix_origin.y);
        U memoryForm0 = src[address0];
        U memoryForm1 = src[address1];
        ((thread T*)thread_elements())[0] = T(memoryForm0);
        ((thread T*)thread_elements())[1] = T(memoryForm1);
      } else if (elements_per_row % 2 != 0) {
        uint address0 = uint(matrix_origin.y) * elements_per_row + uint(matrix_origin.x + 0);
        uint address1 = uint(matrix_origin.y) * elements_per_row + uint(matrix_origin.x + 1);
        U memoryForm0 = src[address0];
        U memoryForm1 = src[address1];
        ((thread T*)thread_elements())[0] = T(memoryForm0);
        ((thread T*)thread_elements())[1] = T(memoryForm1);
      } else {
        auto combinedAddress = uint(matrix_origin.y) * elements_per_row + uint(matrix_origin.x + 0);
        vec<U, 2> memoryForm = *(const device vec<U, 2>*)(src + combinedAddress);
        *(thread_elements()) = vec<T, 2>(memoryForm);
      }
    }

    // WARNING: 'T' must be 'float'.
    METAL_FUNC void load_bfloat(const device bfloat *src, uint elements_per_row, ushort2 matrix_origin, bool transpose_matrix = false) {
      if (transpose_matrix) {
        uint address0 = uint(matrix_origin.x + 0) * elements_per_row + uint(matrix_origin.y);
        uint address1 = uint(matrix_origin.x + 1) * elements_per_row + uint(matrix_origin.y);
        bfloat memoryForm0 = src[address0];
        bfloat memoryForm1 = src[address1];

        bfloat4 registerForm = *(thread bfloat4*)(thread_elements());
        registerForm[1] = memoryForm0;
        registerForm[3] = memoryForm1;
        ((thread bfloat4*)thread_elements())[0] = registerForm;
      } else {
        auto combinedAddress = uint(matrix_origin.y) * elements_per_row + uint(matrix_origin.x + 0);
        bfloat2 memoryForm = *(const device packed_bfloat2*)(src + combinedAddress);

        bfloat4 registerForm = *(thread bfloat4*)(thread_elements());
        ((thread float*)&registerForm)[1] = *(thread float*)(&memoryForm);
        ((thread bfloat*)&registerForm)[1] = memoryForm[0];
        ((thread bfloat4*)thread_elements())[0] = registerForm;
      }
    }

    template <typename U>
    METAL_FUNC void load(const threadgroup U *src, ushort elements_per_row, ushort2 matrix_origin, bool transpose_matrix = false) {
      if (transpose_matrix) {
        ushort address0 = ushort(matrix_origin.x + 0) * elements_per_row + ushort(matrix_origin.y);
        ushort address1 = ushort(matrix_origin.x + 1) * elements_per_row + ushort(matrix_origin.y);
        U memoryForm0 = src[address0];
        U memoryForm1 = src[address1];
        ((thread T*)thread_elements())[0] = T(memoryForm0);
        ((thread T*)thread_elements())[1] = T(memoryForm1);
      } else if (elements_per_row % 2 != 0) {
        ushort address0 = ushort(matrix_origin.y) * elements_per_row + ushort(matrix_origin.x + 0);
        ushort address1 = ushort(matrix_origin.y) * elements_per_row + ushort(matrix_origin.x + 1);
        U memoryForm0 = src[address0];
        U memoryForm1 = src[address1];
        ((thread T*)thread_elements())[0] = T(memoryForm0);
        ((thread T*)thread_elements())[1] = T(memoryForm1);
      } else {
        auto combinedAddress = ushort(matrix_origin.y) * elements_per_row + ushort(matrix_origin.x + 0);
        vec<U, 2> memoryForm = *(const threadgroup vec<U, 2>*)(src + combinedAddress);
        *(thread_elements()) = vec<T, 2>(memoryForm);
      }
    }

    // WARNING: 'T' must be 'float'.
    METAL_FUNC void load_bfloat(const threadgroup bfloat *src, ushort elements_per_row, ushort2 matrix_origin, bool transpose_matrix = false) {
      if (transpose_matrix) {
        ushort address0 = ushort(matrix_origin.x + 0) * elements_per_row + ushort(matrix_origin.y);
        ushort address1 = ushort(matrix_origin.x + 1) * elements_per_row + ushort(matrix_origin.y);
        bfloat memoryForm0 = src[address0];
        bfloat memoryForm1 = src[address1];

        bfloat4 registerForm = *(thread bfloat4*)(thread_elements());
        registerForm[1] = memoryForm0;
        registerForm[3] = memoryForm1;
        ((thread bfloat4*)thread_elements())[0] = registerForm;
      } else {
        auto combinedAddress = ushort(matrix_origin.y) * elements_per_row + ushort(matrix_origin.x + 0);
        bfloat2 memoryForm = *(const threadgroup packed_bfloat2*)(src + combinedAddress);

        bfloat4 registerForm = *(thread bfloat4*)(thread_elements());
        ((thread float*)&registerForm)[1] = *(thread float*)(&memoryForm);
        ((thread bfloat*)&registerForm)[1] = memoryForm[0];
        ((thread bfloat4*)thread_elements())[0] = registerForm;
      }
    }

    template <typename U>
    METAL_FUNC void store(device U *dst, uint elements_per_row, ushort2 matrix_origin, bool transpose_matrix = false) {
      if (transpose_matrix) {
        uint address0 = uint(matrix_origin.x + 0) * elements_per_row + uint(matrix_origin.y);
        uint address1 = uint(matrix_origin.x + 1) * elements_per_row + uint(matrix_origin.y);
        T registerForm0 = ((thread T*)thread_elements())[0];
        T registerForm1 = ((thread T*)thread_elements())[1];
        dst[address0] = U(registerForm0);
        dst[address1] = U(registerForm1);
      } else if (elements_per_row % 2 != 0) {
        uint address0 = uint(matrix_origin.y) * elements_per_row + uint(matrix_origin.x + 0);
        uint address1 = uint(matrix_origin.y) * elements_per_row + uint(matrix_origin.x + 1);
        T registerForm0 = ((thread T*)thread_elements())[0];
        T registerForm1 = ((thread T*)thread_elements())[1];
        dst[address0] = U(registerForm0);
        dst[address1] = U(registerForm1);
      } else {
        auto combinedAddress = uint(matrix_origin.y) * elements_per_row + uint(matrix_origin.x + 0);
        vec<T, 2> registerForm = *(thread_elements());
        *(device vec<U, 2>*)(dst + combinedAddress) = vec<U, 2>(registerForm);
      }
    }

    // WARNING: 'T' must be 'float'.
    METAL_FUNC void store_bfloat(device bfloat *dst, uint elements_per_row, ushort2 matrix_origin, bool transpose_matrix = false) {
      if (transpose_matrix) {
        uint address0 = uint(matrix_origin.x + 0) * elements_per_row + uint(matrix_origin.y);
        uint address1 = uint(matrix_origin.x + 1) * elements_per_row + uint(matrix_origin.y);
        bfloat4 registerForm = *(thread bfloat4*)(thread_elements());
        registerForm[2] = registerForm[1];
        dst[address0] = registerForm[2];
        dst[address1] = registerForm[3];
      } else if (elements_per_row % 2 != 0) {
        uint address0 = uint(matrix_origin.y) * elements_per_row + uint(matrix_origin.x + 0);
        uint address1 = uint(matrix_origin.y) * elements_per_row + uint(matrix_origin.x + 1);
        bfloat4 registerForm = *(thread bfloat4*)(thread_elements());
        registerForm[2] = registerForm[1];
        dst[address0] = registerForm[2];
        dst[address1] = registerForm[3];
      } else {
        auto combinedAddress = uint(matrix_origin.y) * elements_per_row + uint(matrix_origin.x + 0);
        bfloat4 registerForm = *(thread bfloat4*)(thread_elements());
        registerForm[2] = registerForm[1];
        float memoryForm = ((thread float*)&registerForm)[1];
        *(device float*)(dst + combinedAddress) = memoryForm;
      }
    }

    template <typename U>
    METAL_FUNC void store(threadgroup U *dst, ushort elements_per_row, ushort2 matrix_origin, bool transpose_matrix = false) {
      if (transpose_matrix) {
        ushort address0 = ushort(matrix_origin.x + 0) * elements_per_row + ushort(matrix_origin.y);
        ushort address1 = ushort(matrix_origin.x + 1) * elements_per_row + ushort(matrix_origin.y);
        T registerForm0 = ((thread T*)thread_elements())[0];
        T registerForm1 = ((thread T*)thread_elements())[1];
        dst[address0] = U(registerForm0);
        dst[address1] = U(registerForm1);
      } else if (elements_per_row % 2 != 0) {
        ushort address0 = ushort(matrix_origin.y) * elements_per_row + ushort(matrix_origin.x + 0);
        ushort address1 = ushort(matrix_origin.y) * elements_per_row + ushort(matrix_origin.x + 1);
        T registerForm0 = ((thread T*)thread_elements())[0];
        T registerForm1 = ((thread T*)thread_elements())[1];
        dst[address0] = U(registerForm0);
        dst[address1] = U(registerForm1);
      } else {
        auto combinedAddress = ushort(matrix_origin.y) * elements_per_row + ushort(matrix_origin.x + 0);
        vec<T, 2> registerForm = *(thread_elements());
        *(threadgroup vec<U, 2>*)(dst + combinedAddress) = vec<U, 2>(registerForm);
      }
    }

    // WARNING: 'T' must be 'float'.
    METAL_FUNC void store_bfloat(threadgroup bfloat *dst, ushort elements_per_row, ushort2 matrix_origin, bool transpose_matrix = false) {
      if (transpose_matrix) {
        ushort address0 = ushort(matrix_origin.x + 0) * elements_per_row + ushort(matrix_origin.y);
        ushort address1 = ushort(matrix_origin.x + 1) * elements_per_row + ushort(matrix_origin.y);
        bfloat4 registerForm = *(thread bfloat4*)(thread_elements());
        registerForm[2] = registerForm[1];
        dst[address0] = registerForm[2];
        dst[address1] = registerForm[3];
      } else if (elements_per_row % 2 != 0) {
        ushort address0 = ushort(matrix_origin.y) * elements_per_row + ushort(matrix_origin.x + 0);
        ushort address1 = ushort(matrix_origin.y) * elements_per_row + ushort(matrix_origin.x + 1);
        bfloat4 registerForm = *(thread bfloat4*)(thread_elements());
        registerForm[2] = registerForm[1];
        dst[address0] = registerForm[2];
        dst[address1] = registerForm[3];
      } else {
        auto combinedAddress = ushort(matrix_origin.y) * elements_per_row + ushort(matrix_origin.x + 0);
        bfloat4 registerForm = *(thread bfloat4*)(thread_elements());
        registerForm[2] = registerForm[1];
        float memoryForm = ((thread float*)&registerForm)[1];
        *(threadgroup float*)(dst + combinedAddress) = memoryForm;
      }
    }

    template <typename U, typename V>
    METAL_FUNC void multiply(simdgroup_matrix_storage<U> a, simdgroup_matrix_storage<V> b, bool accumulate = true) {
      if (!accumulate) {
        *(thread_elements()) = vec<T, 2>(0);
      }
      t = __metal_simdgroup_matrix_8x8_multiply_accumulate(a.t, b.t, t, typename simdgroup_matrix_storage<T>::storage_type());
    }
  };
} // namespace metal
#pragma METAL internals : disable

#endif
