/* A vamana graph index for large groups of points */
#pragma once

#include "virtual_index.h"
#include "small_index.h"

#include "../utils/point_range.h"
#include "../utils/graph.h"
#include "../utils/beamSearch.h"
#include "../vamana/index.h"

#include <algorithm>
#include <iostream>

// limit, degree, alpha
const BuildParams default_build_params = BuildParams(200, 32, 1.175);

// k, beam size, cut, limit, degree limit
const QueryParams default_query_params = QueryParams(100, 150, 0.9, 1000, 100);

template<typename T, typename Point>
struct VamanaIndex : public VirtualIndex<T, Point> {
    NaiveIndex<T, Point> naive_index;
    Graph<index_type> graph;

    VamanaIndex() = default;

    void fit(PointRange<T, Point>& points,
             parlay::sequence<float>& timestamps,
             parlay::sequence<index_type>& indices) override {
        naive_index.fit(points, timestamps, indices);
    }

    void fit(PointRange<T, Point>& points,
             parlay::sequence<float>& timestamps) override {
        auto indices = parlay::tabulate(points.size(), [](index_type i) { return i; });

        fit(points, timestamps, indices);
    }

    void knn(Point& query, index_type* out, size_t k) const override {
        naive_index.knn(query, out, k);
    }

    void range_knn(Point& query, index_type* out, std::pair<float, float> endpoints, size_t k) const override {
        naive_index.range_knn(query, out, endpoints, k);
    }
};
