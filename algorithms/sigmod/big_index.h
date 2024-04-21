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

    void train_graph()

    void fit(PointRange<T, Point>& points,
             parlay::sequence<float>& timestamps,
             parlay::sequence<index_type>& indices) override {
        naive_index.fit(points, timestamps, indices);

        G = Graph<index_type>(points.size());

        knn_index<Point, SubsetPointRange<T, Point, PointRange<T, Point>, uint32_t>> I(default_build_params);

        stats<index_type> BuildStats(this->points.size());

        I.build_index(G, naive_index.pr, BuildStats);
    }

    void fit(PointRange<T, Point>& points,
             parlay::sequence<float>& timestamps) override {
        auto indices = parlay::tabulate(points.size(), [](index_type i) { return i; });

        fit(points, timestamps, indices);
    }

    void knn(Point& query, index_type* out, size_t k) const override {
        auto [pairElts, dist_cmps] = beam_search<Point, SubsetPointRange<T, Point, PointRange<T, Point>, uint32_t>, index_type>(query, G, naive_index.pr, 0,default_query_params);

        for (size_t i = 0; i < k; i++) {
            out[i] = naive_index.pr.real_index(pairElts[i].second);
        }
    }

    void range_knn(Point& query, index_type* out, std::pair<float, float> endpoints, size_t k) const override {
        naive_index.range_knn(query, out, endpoints, k);
    }
};
