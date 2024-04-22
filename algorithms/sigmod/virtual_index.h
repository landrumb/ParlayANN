#pragma once

#include "../utils/point_range.h"
#include "../utils/types.h"

#include <cstdint>
#include <utility>

using index_type = uint32_t;

/* This is a virtual parent class for component indices. 

Anything expected to be used to index the points associated with a label should subclass this, */
template <typename T, typename Point>
class VirtualIndex {
    public:
    virtual void fit(PointRange<T, Point>& points,
             parlay::sequence<float>& timestamps,
             parlay::sequence<index_type>& indices) = 0;

    virtual void fit(PointRange<T, Point>& points,
             parlay::sequence<float>& timestamps) = 0;
    /* Unfiltered knn query, where out is a pointer to where the indices of the neighbors should be written
    (this saves us a copy and some allocations over returning a sequence) */
    virtual void knn(Point& query, index_type* out, size_t k)  = 0;
    /* Range-filtered knn query */
    virtual void range_knn(Point& query, index_type* out, std::pair<float, float> endpoints, size_t k) = 0;
    virtual void _range_knn(Point& query, index_type* out, float *dists, index_type left_end, index_type right_end, float min_time, float max_time, size_t k) = 0;

};
