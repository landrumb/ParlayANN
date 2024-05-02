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
    // virtual size_t _range_knn(Point& query, index_type* out, float *dists, index_type left_end, index_type right_end, float min_time, float max_time, size_t k) = 0;
    /* A method for handling batches of knn queries
    
    queries is a pointer to an array of aligned values of points that are to be queried
    out is a pointer to an array of pointers where the results will be stored for each query
    k is the number of nearest neighbors to find
    num_queries is the number of query vectors in the queries array
     */
    virtual void batch_knn(T* queries, index_type** out, size_t k, size_t num_queries, bool parallel = true) {
        if (parallel) {
            parlay::parallel_for(0, num_queries, [&] (size_t i) {
                Point query(queries + i * aligned_dims(), dims(), aligned_dims(), i);
                knn(query, *(out + i), k);
            });
        } else {
            for (size_t i = 0; i < num_queries; i++) {
                Point query(queries + i * aligned_dims(), dims(), aligned_dims(), i);
                knn(query, *(out + i), k);
            }
        }
    }

    virtual size_t size() const = 0;
    virtual size_t dims() const = 0;
    virtual size_t aligned_dims() const = 0;
};
