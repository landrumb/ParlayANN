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
#include <cstring>

// limit, degree, alpha
//BuildParams default_build_params = BuildParams(200, 32, 1.175);
BuildParams default_build_params = BuildParams(500, 64, 1.175);

// k, beam size, cut, limit, degree limit
//QueryParams default_query_params = QueryParams(100, 500, 0.9, 1000, 100);
QueryParams default_query_params = QueryParams(100, 500, 1.35, 10000000, 100);

const float exhaustive_fallback_cutoff = 0.25;

const float overretrieval_cutoff = 0.5;

void set_default_build_params(long R, long L, double alpha) {
    default_build_params = BuildParams(R, L, alpha);
}

void set_default_query_params(long k, long beamSize, double cut, long limit, long degree_limit) {
    default_query_params = QueryParams(k, beamSize, cut, limit, degree_limit);
}

// Any graphs smaller than this are replaced with naive indexes
const uint32_t window_min_graph_size = 100;

template<typename T, typename Point>
struct VamanaIndex : public VirtualIndex<T, Point> {
    NaiveIndex<T, Point> naive_index;
    Graph<index_type> G;

    VirtualIndex<T, Point> *left_child;
    VirtualIndex<T, Point> *right_child;
    uint32_t mid;

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

    int _range_knn(Point& query, index_type* out, float *dists, uint32_t left_end, uint32_t right_end, float min_time, float max_time, size_t k) override {
        float range_percentage = (float)(right - left) / g.size();

        if (range_percentage > overretrieval_cutoff) {
            QueryParams qp = default_query_params;
            qp.k = qp.beamSize;
            //qp.limit = static_cast<int>(qp.limit / (endpoints.second - endpoints.first));
 
            auto [pairElts, dist_cmps] = beam_search<Point, SubsetPointRange<T, Point, PointRange<T, Point>, uint32_t>, index_type>(query, G, naive_index.pr, 0, qp);

            auto frontier = pairElts.first;

            // filter to the points which are in the range
            int found = 0;
            for (size_t i = 0; i < frontier.size() && found < k; i++) {
                if (naive_index.timestamps[frontier[i].first] >= left_time && naive_index.timestamps[frontier[i].first] <= right_time) {
                    out[found] = naive_index.pr.real_index(frontier[i].first);
                    found++;
                }
            }

            if (found < k) {
                std::cout << "Warning: not enough points in range" << std::endl;
            }
        } else {
            if (left_end < mid && right_end > mid) {
                auto left_out = parlay::sequence<index_type>::uninitialized(k);
                auto left_dists = parlay::sequence<float>::uninitialized(k);
                auto right_out = parlay::sequence<index_type>::uninitialized(k);
                auto right_dists = parlay::sequence<float>::uninitialized(k);

                int left_found, right_found;

                parlay::par_do([&] () {
                    left_found = left_child->_range_knn(query, &left_out[0], &left_dists[0], left_end, mid, min_time, max_time, k);
                }, [&] () {
                    right_found = right_child->_range_knn(query, &right_out[0], &right_dists[0], 0, right_end - mid, min_time, max_time, k);
                });

                int i = 0, j = 0, found = 0;
                while (i < left_found && j < right_found && found < k) {
                    if (left_dists[i] < right_dists[j]) {
                        out[found] = left_out[i];
                        dists[found++] = left_dists[i++];
                    }
                    else {
                        out[found] = right_out[j];
                        dists[found++] = right_dists[j++];
                    }
                }
                if (i == left_found) {
                    int extra = std::min(right_found - j, k - found);
                    std::memcpy(&out[found], &right_out[j], extra);
                    std::memcpy(&dists[found], &right_dists[j], extra);
                    found += extra;
                }
                else if (j == right_found) {
                    int extra = std::min(left_found - i, k - found);
                    std::memcpy(&out[found], &left_out[i], extra);
                    std::memcpy(&dists[found], &left_dists[i], extra);
                    found += extra;
                }
                return found;
            }
            else if (left_end < mid) {
                return left_child->_range_knn(query, out, dists, left_end, right_end, min_time, max_time, k);
            }
            else {
                return right_child->_range_knn(query, out, dists, left_end - mid, right_end - mid, min_time, max_time, k);
            }
        }
    }

    void range_knn(Point& query, index_type* out, std::pair<float, float> endpoints, size_t k) override {
        float time_range = endpoints.second - endpoints.first;

        if (time_range < exhaustive_fallback_cutoff) {
            naive_index.range_knn(query, out, endpoints, k);
            return;
        } else { // otherwise we use overretrieval
            QueryParams qp = default_query_params;
            qp.k = qp.beamSize;
            qp.limit = static_cast<int>(qp.limit / (endpoints.second - endpoints.first));
 
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

    /*void range_knn(Point& query, index_type* out, std::pair<float, float> endpoints, size_t k) override {
        // we assume that the distribution of timestamps is uniform, and the distance between the endpoints is some good approximation of the selectivity
        float time_range = endpoints.second - endpoints.first;

        if (time_range < exhaustive_fallback_cutoff) {
            naive_index.range_knn(query, out, endpoints, k);
            return;
        } else { // otherwise we use overretrieval
            QueryParams qp = default_query_params;
            qp.k = qp.beamSize;
            qp.limit = static_cast<int>(qp.limit / (endpoints.second - endpoints.first));
            //qp.k = static_cast<int>((qp.k / (endpoints.second - endpoints.first)) * 1.44);
            //qp.beamSize = static_cast<int>((qp.beamSize / (endpoints.second - endpoints.first)) * 1.44);
            //qp.limit = static_cast<int>((qp.limit / (endpoints.second - endpoints.first)) * 1.44);
 
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
    }*/
};
