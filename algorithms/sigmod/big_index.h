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
//BuildParams default_build_params = BuildParams(200, 32, 1.175);
BuildParams default_build_params = BuildParams(500, 64, 1.175);

// k, beam size, cut, limit, degree limit
//QueryParams default_query_params = QueryParams(100, 500, 0.9, 1000, 100);
QueryParams default_query_params = QueryParams(100, 500, 1.35, 10000000, 100);

const float exhaustive_fallback_cutoff = 0.25;

void set_default_build_params(long R, long L, double alpha) {
    default_build_params = BuildParams(R, L, alpha);
}

void set_default_query_params(long k, long beamSize, double cut, long limit, long degree_limit) {
    default_query_params = QueryParams(k, beamSize, cut, limit, degree_limit);
}

template<typename T, typename Point>
struct VamanaIndex : public VirtualIndex<T, Point> {
    NaiveIndex<T, Point> naive_index;
    Graph<index_type> G;

    parlay::sequence<parlay::sequence<VirtualIndex<T, Point>*>> window_indices;
    parlay::sequence<parlay::sequence<std::pair<float, float>>> window_bounds;

    VamanaIndex() = default;

    void fit(PointRange<T, Point>& points,
             parlay::sequence<float>& timestamps,
             parlay::sequence<index_type>& indices) override {
        naive_index.fit(points, timestamps, indices);

        G = Graph<index_type>(default_build_params.R, indices.size());

        knn_index<Point, SubsetPointRange<T, Point, PointRange<T, Point>, uint32_t>, uint32_t> I(default_build_params);

        stats<index_type> BuildStats(points.size());

        I.build_index(G, naive_index.pr, BuildStats);
    }

    void fit(PointRange<T, Point>& points,
             parlay::sequence<float>& timestamps) override {
        auto indices = parlay::tabulate(points.size(), [](index_type i) { return i; });

        fit(points, timestamps, indices);
    }

    void knn(Point& query, index_type* out, size_t k) override {
        auto [pairElts, dist_cmps] = beam_search<Point, SubsetPointRange<T, Point, PointRange<T, Point>, uint32_t>, index_type>(query, G, naive_index.pr, 0,default_query_params);

        auto frontier = pairElts.first;

        for (size_t i = 0; i < k; i++) {
            out[i] = naive_index.pr.real_index(frontier[i].first);
        }
    }

    void range_knn(Point& query, index_type* out, std::pair<float, float> endpoints, size_t k) override {
        // we assume that the distribution of timestamps is uniform, and the distance between the endpoints is some good approximation of the selectivity
        float time_range = endpoints.second - endpoints.first;

        if (time_range < exhaustive_fallback_cutoff) {
            naive_index.range_knn(query, out, endpoints, k);
            return;
        } else { // otherwise we use overretrieval
            QueryParams qp = default_query_params;
            qp.k = qp.beamSize;
            qp.limit = static_cast<int>(qp.limit / (endpoints.second - endpoints.first));
            /*qp.k = static_cast<int>((qp.k / (endpoints.second - endpoints.first)) * 1.44);
            qp.beamSize = static_cast<int>((qp.beamSize / (endpoints.second - endpoints.first)) * 1.44);
            qp.limit = static_cast<int>((qp.limit / (endpoints.second - endpoints.first)) * 1.44);*/
 
            auto [pairElts, dist_cmps] = beam_search<Point, SubsetPointRange<T, Point, PointRange<T, Point>, uint32_t>, index_type>(query, G, naive_index.pr, 0, qp);

            auto frontier = pairElts.first;

            // filter to the points which are in the range
            size_t found = 0;
            for (size_t i = 0; i < frontier.size() && found < k; i++) {
                if (naive_index.timestamps[frontier[i].first] >= endpoints.first && naive_index.timestamps[frontier[i].first] <= endpoints.second) {
                    out[found] = naive_index.pr.real_index(frontier[i].first);
                    found++;
                }
            }

            if (found < k) {
                std::cout << "Warning: not enough points in range" << std::endl;
            }
        }
    }
};
