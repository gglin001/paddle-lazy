#pragma once

#include <glog/logging.h>
#include <paddle/phi/core/allocator.h>

class LazyAllocator : public phi::Allocator {
 public:
  // singleton
  static LazyAllocator* Instance() {
    static auto instance = std::make_unique<LazyAllocator>();
    return instance.get();
  }

  static void Deleter(phi::Allocation* allocation) { delete allocation; }

  AllocationPtr Allocate(size_t bytes_size) override {
    LOG(ERROR) << "LazyAllocator trying to allocate " << bytes_size;
    auto alloc = new phi::Allocation(nullptr, 0, phi::IPUPlace());
    return AllocationPtr(alloc, LazyAllocator::Deleter);
  }
};
