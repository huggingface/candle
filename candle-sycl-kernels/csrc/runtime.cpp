// Host-side SYCL runtime: queue, USM, memcpy, device info.
#include "common.hpp"
#include <cstring>
#include <vector>

using sycl::queue;

namespace {
std::vector<sycl::device> gpu_devices() {
  std::vector<sycl::device> out;
  for (const auto &p : sycl::platform::get_platforms()) {
    for (const auto &d : p.get_devices(sycl::info::device_type::gpu)) {
      out.push_back(d);
    }
  }
  return out;
}
} // namespace

extern "C" {

int candle_sycl_device_count(void) {
  try {
    return (int)gpu_devices().size();
  } catch (...) {
    return 0;
  }
}

CandleSyclQueue *candle_sycl_queue_new(int ordinal) {
  try {
    auto devs = gpu_devices();
    if (ordinal < 0 || (size_t)ordinal >= devs.size()) {
      return nullptr;
    }
    return new CandleSyclQueue(devs[ordinal]);
  } catch (...) {
    return nullptr;
  }
}

void candle_sycl_queue_free(CandleSyclQueue *q) { delete q; }

void *candle_sycl_queue_native(CandleSyclQueue *q) { return &q->q; }

int candle_sycl_synchronize(CandleSyclQueue *q) {
  try {
    q->q.wait_and_throw();
    return CANDLE_SYCL_OK;
  } catch (...) {
    return CANDLE_SYCL_ERR_EXCEPTION;
  }
}

int candle_sycl_device_info(CandleSyclQueue *q, CandleSyclDeviceInfo *out) {
  try {
    auto d = q->q.get_device();
    auto name = d.get_info<sycl::info::device::name>();
    std::memset(out, 0, sizeof(*out));
    std::strncpy(out->name, name.c_str(), sizeof(out->name) - 1);
    out->global_mem_bytes = d.get_info<sycl::info::device::global_mem_size>();
    out->max_compute_units = d.get_info<sycl::info::device::max_compute_units>();
    out->max_clock_khz = d.get_info<sycl::info::device::max_clock_frequency>() * 1000u;
    out->is_integrated = d.get_info<sycl::info::device::host_unified_memory>() ? 1 : 0;
    out->supports_fp16 = d.has(sycl::aspect::fp16) ? 1 : 0;
    out->supports_fp64 = d.has(sycl::aspect::fp64) ? 1 : 0;
    return CANDLE_SYCL_OK;
  } catch (...) {
    return CANDLE_SYCL_ERR_EXCEPTION;
  }
}

void *candle_sycl_malloc(CandleSyclQueue *q, size_t bytes) {
  try {
    if (bytes == 0) {
      bytes = 1;
    }
    return sycl::malloc_device(bytes, q->q);
  } catch (...) {
    return nullptr;
  }
}

void candle_sycl_free(CandleSyclQueue *q, void *ptr) {
  if (ptr) {
    sycl::free(ptr, q->q);
  }
}

int candle_sycl_memcpy_htod(CandleSyclQueue *q, void *dst, const void *src, size_t bytes) {
  try {
    if (bytes) q->q.memcpy(dst, src, bytes).wait_and_throw();
    return CANDLE_SYCL_OK;
  } catch (...) {
    return CANDLE_SYCL_ERR_EXCEPTION;
  }
}
int candle_sycl_memcpy_dtoh(CandleSyclQueue *q, void *dst, const void *src, size_t bytes) {
  return candle_sycl_memcpy_htod(q, dst, src, bytes);
}
int candle_sycl_memcpy_dtod(CandleSyclQueue *q, void *dst, const void *src, size_t bytes) {
  return candle_sycl_memcpy_htod(q, dst, src, bytes);
}

int candle_sycl_memset(CandleSyclQueue *q, void *dst, int value, size_t bytes) {
  try {
    if (bytes) q->q.memset(dst, value, bytes).wait_and_throw();
    return CANDLE_SYCL_OK;
  } catch (...) {
    return CANDLE_SYCL_ERR_EXCEPTION;
  }
}

} // extern "C"
