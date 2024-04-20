/* A naive exhaustive search index */

#include "virtual_index.h"

#include "../utils/point_range.h"

#include <algorithm>
#include <iostream>

using index_type = uint32_t;

template <typename T, typename Point>
struct NaiveIndex : public VirtualIndex<T, Point> {
    SubsetPointRange<T, Point, PointRange<T, Point>, uint32_t> pr; // this will need to change if we collect copies of the vectors
    parlay::sequence<float> timestamps;

    NaiveIndex() = default;

    void fit(PointRange<T, Point>& points,
             parlay::sequence<float>& timestamps,
             parlay::sequence<index_type>& indices) override {
        // before creating the pointrange we want to argsort the indices by timestamp
        parlay::sequence<index_type> sorted_subset_indices = parlay::sequence<index_type>(indices.size());

        std::sort(sorted_subset_indices.begin(), sorted_subset_indices.end(),
                  [&timestamps](index_type i, index_type j) {
                      return timestamps[i] < timestamps[j];
                  });

        parlay::sequence<index_type> sorted_indices(indices.size());
        parlay::sequence<float> sorted_timestamps(indices.size());

        for (size_t i = 0; i < indices.size(); i++) {
            sorted_indices[i] = indices[sorted_subset_indices[i]];
            sorted_timestamps[i] = timestamps[sorted_subset_indices[i]];
        }

        pr = SubsetPointRange<T, Point, PointRange<T, Point>, uint32_t>(points, sorted_indices);
        this->timestamps = std::move(sorted_timestamps);
    }

    void fit(PointRange<T, Point>& points,
             parlay::sequence<float>& timestamps) override {
        auto indices = parlay::tabulate(points.size(), [](index_type i) { return i; });

        fit(points, timestamps, indices);
    }

    /* internal method to centralize exhaustive search logic
    
    range is of the form (start, length) */
    inline void _index_range_knn(Point& query, index_type* out, size_t k, std::pair<index_type, index_type> range) const {
        // for the sake of avoiding overhead from nested parallelism, we will compute distances serially
        parlay::sequence<std::pair<float, index_type>> distances(range.second);

        for (index_type i = range.first; i < range.first + range.second; i++) {
            distances[i] = std::make_pair(query.distance(pr[i]), i);
        }

        std::sort(distances.begin(), distances.end()); // technically a top k = 100 but we'll just sort the whole thing

        for (size_t i = 0; i < k; i++) {
            out[i] = pr.real_index(distances[i].second);
        }
    }

    void knn(Point& query, index_type* out, size_t k) const override {
        _index_range_knn(query, out, k, std::make_pair(0, pr.size()));
    }

    void range_knn(Point& query, index_type* out, std::pair<float, float> endpoints, size_t k) const override {
        // we will need to do a binary search to find the start and end of the range
        index_type start = 0; // start is the index of the first element geq endpoints.first
        index_type end = pr.size(); // end is the index of the last element leq endpoints.second

        index_type l = 0;
        index_type r = pr.size();

        while (l < r) {
            index_type m = (l + r) / 2;
            if (this->timestamps[m] < endpoints.first) {
                l = m + 1;
            } else {
                r = m;
            }
        }
        start = l;

        l = 0;
        r = pr.size();

        while (l < r) {
            index_type m = (l + r) / 2;
            if (this->timestamps[m] <= endpoints.second) {
                l = m + 1;
            } else {
                r = m;
            }
        }
        end = l;

        _index_range_knn(query, out, k, std::make_pair(start, end - start));
    }

};