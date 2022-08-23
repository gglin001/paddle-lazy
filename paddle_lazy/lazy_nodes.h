#pragma once

#include "paddle/phi/common/int_array.h"
#include "paddle/phi/common/scalar.h"
#include "paddle_lazy/lazy_nodes_autogen.h"

namespace phi {

class FullLazyNode : public LazyNode {
 public:
  FullLazyNode(const IntArray& shape,
               const Scalar& value,
               DataType dtype = DataType::FLOAT32,
               const Place& place = CPUPlace())
      : shape(shape), value(value), dtype(dtype), place(place) {
    op_type = "full";
  }

  IntArray shape;
  Scalar value;
  DataType dtype;
  Place place;
};

}  // namespace phi
