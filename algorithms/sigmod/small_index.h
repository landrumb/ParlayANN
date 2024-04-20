/* A naive exhaustive search index */

#include "virtual_index.h"

#include "../utils/point_range.h"

#include <algorithm>
#include <iostream>

using index_type = uint32_t;

struct NaiveIndex : public VirtualIndex<float> {
  parlay::sequence<float> values;
  parlay::sequence<index_type> labels;
  parlay::sequence<float> timestamps;
  parlay::sequence<index_type> indices;

}