#pragma once

#include "../utils/point_range.h"
#include "../utils/types.h"

#include <cstdint>
#include <utility>

using index_type = uint32_t;

/* This is a virtual parent class for component indices. 

Anything expected to be used to index the points associated with a label should subclass this, */
template <typename Point>
class VirtualIndex {
// virtual void fit(PointRange<Point>& points, parlay::sequence<index_type>& labels, parlay::sequence<float>& timestamps, parlay::sequence<index_type>& indices) = 0;
/* Unfiltered knn query, where out is a pointer to where the indices of the neighbors should be written
(this saves us a copy and some allocations over returning a sequence) */
virtual void knn(Point& query, index_type* out, size_t k) const = 0;
/* Range-filtered knn query */
virtual void range_knn(Point& query, index_type* out, std::pair<float, float> endpoints, size_t k) const = 0;

};
