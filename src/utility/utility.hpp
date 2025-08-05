#ifdef USE_CUDA
#pragma once
#include <c10/cuda/CUDACachingAllocator.h>
#include <memory>
#include <stdexcept>
#include <torch/torch.h>
class GpuBuffer {
public:
    explicit GpuBuffer(size_t bytes) : bytes_(bytes), ptr_(nullptr) {
        ptr_ = c10::cuda::CUDACachingAllocator::raw_alloc(bytes_);
        if (!ptr_) {
            throw std::runtime_error("GPU allocation failed");
        }
    }
    GpuBuffer(const GpuBuffer &) = delete;
    GpuBuffer &operator=(const GpuBuffer &) = delete;

    GpuBuffer(GpuBuffer &&other) noexcept : bytes_(other.bytes_), ptr_(other.ptr_) {
        other.ptr_ = nullptr;
        other.bytes_ = 0;
    }

    GpuBuffer &operator=(GpuBuffer &&other) noexcept {
        if (this != &other) {
            release();
            ptr_ = other.ptr_;
            bytes_ = other.bytes_;
            other.ptr_ = nullptr;
            other.bytes_ = 0;
        }
        return *this;
    }

    ~GpuBuffer() { release(); }

    void *data() const { return ptr_; }
    size_t size() const { return bytes_; }

private:
    void release() {
        if (ptr_) {
            c10::cuda::CUDACachingAllocator::raw_delete(ptr_);
            ptr_ = nullptr;
        }
    }

    size_t bytes_;
    void *ptr_;
};

#endif