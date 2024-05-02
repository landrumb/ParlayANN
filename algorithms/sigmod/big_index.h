/* A vamana graph index for large groups of points */
#pragma once

#include "virtual_index.h"
#include "small_index.h"
#include "filter_ivf_index.h"

#include "../utils/point_range.h"
#include "../utils/graph.h"
#include "../utils/beamSearch.h"
#include "../vamana/index.h"

#include <algorithm>
#include <iostream>
#include <cstring>
#include <cmath>

// limit, degree, alpha
//BuildParams default_build_params = BuildParams(200, 32, 1.175);
BuildParams default_build_params = BuildParams(500, 64, 1.175);

// k, beam size, cut, limit, degree limit
//QueryParams default_query_params = QueryParams(100, 500, 0.9, 1000, 100);
QueryParams default_query_params = QueryParams(100, 500, 1.35, 10000000, 100);

QueryParams default_overretrieval_params = QueryParams(500, 500, 1.35, 10000000, 100);

float exhaustive_fallback_cutoff = 0.25;
size_t tiny_cutoff = 2000;

size_t min_size = 10'000;

const float overretrieval_cutoff = 0.5;

void set_default_build_params(long R, long L, double alpha) {
    default_build_params = BuildParams(R, L, alpha);
}

void set_default_query_params(long k, long beamSize, double cut, long limit, long degree_limit) {
    default_query_params = QueryParams(k, beamSize, cut, limit, degree_limit);
}

void set_exhaustive_fallback_cutoff(float cutoff) {
    exhaustive_fallback_cutoff = cutoff;
}

void set_min_size(size_t size) {
    min_size = size;
}

/* worth noting that this is not robust to duplicates */
void update_frontier(parlay::sequence<std::pair<float, index_type>>& frontier, parlay::sequence<std::pair<float, index_type>>& adds, size_t k) {
    frontier.append(adds);

    std::sort(frontier.begin(), frontier.end(), [](auto a, auto b) { return a.first < b.first; });
    
    if (frontier.size() > k) {
        frontier.resize(k);
    }
}

template<typename T, typename Point>
struct VamanaIndex : public VirtualIndex<T, Point> {
    NaiveIndex<T, Point> naive_index;
    Graph<index_type> G;

    /*std::unique_ptr<VamanaIndex<T, Point>> left = nullptr;
    std::unique_ptr<VamanaIndex<T, Point>> right = nullptr;
    uint32_t mid;*/

    Filter_IVF<T, Point> ivf;

    VamanaIndex() = default;

    void fit(PointRange<T, Point>& points,
             parlay::sequence<float>& timestamps,
             parlay::sequence<index_type>& indices) override {
        naive_index.fit(points, timestamps, indices);

        G = Graph<index_type>(default_build_params.R, indices.size());

        knn_index<Point, SubsetPointRange<T, Point, PointRange<T, Point>, uint32_t>, uint32_t> I(default_build_params);

        stats<index_type> BuildStats(points.size());

        I.build_index(G, naive_index.pr, BuildStats);

        /*if (points.size() >= min_size * 2) {
            parlay::sequence<index_type> left_indices(indices.begin(), indices.begin() + indices.size() / 2);
            parlay::sequence<index_type> right_indices(indices.begin() + indices.size() / 2, indices.end());

            parlay::sequence<float> left_timestamps(timestamps.begin(), timestamps.begin() + timestamps.size() / 2);
            parlay::sequence<float> right_timestamps(timestamps.begin() + timestamps.size() / 2, timestamps.end());

            left = std::make_unique<VamanaIndex<T, Point>>();
            right = std::make_unique<VamanaIndex<T, Point>>();

            NaiveIndex<T, Point> left_ni, right_ni;

            left_ni.copyless_fit(points, left_timestamps, left_indices);
            right_ni.copyless_fit(points, right_timestamps, right_indices);

            left->fit(left_ni);
            right->fit(right_ni);

            mid = indices.size() / 2;
        }
        
        size_t n_clusters = static_cast<size_t>(indices.size() / min_size);
        ivf = Filter_IVF<T, Point>(naive_index.pr, timestamps, n_clusters);*/
    }

    void fit(PointRange<T, Point>& points,
             parlay::sequence<float>& timestamps) override {
        auto indices = parlay::tabulate(points.size(), [](index_type i) { return i; });

        fit(points, timestamps, indices);
    }

    void fit(NaiveIndex<T, Point>& ni) {
        naive_index = std::move(ni);

        G = Graph<index_type>(default_build_params.R, naive_index.timestamps.size());

        knn_index<Point, SubsetPointRange<T, Point, PointRange<T, Point>, uint32_t>, uint32_t> I(default_build_params);

        stats<index_type> BuildStats(naive_index.timestamps.size());

        I.build_index(G, naive_index.pr, BuildStats);

        
    }

    void knn(Point& query, index_type* out, size_t k) override {
        auto [pairElts, dist_cmps] = beam_search<Point, SubsetPointRange<T, Point, PointRange<T, Point>, uint32_t>, index_type>(query, G, naive_index.pr, 0,default_query_params);

        auto frontier = pairElts.first;

        for (size_t i = 0; i < k; i++) {
            out[i] = naive_index.pr.real_index(frontier[i].first);
        }
    }

    /* 
    // returns the number of nearest neighbors found
    // out and dists are for outputting the indexes and distances of the k-closest
    // [left_end, right_end) indicates the range of indices within the desired timestamps
    // min_time and max_time are the timestamps of the queried range (inclusive)
    size_t _range_knn(Point& query, index_type* out, float *dists, index_type left_end, index_type right_end, float min_time, float max_time, size_t k) override {
        float range_percentage = (float)(right_end - left_end) / G.size();

        // if the range is large, do overretrieval instead of recursing to children
        if (range_percentage > overretrieval_cutoff) {
            QueryParams qp = default_query_params;
            qp.k = qp.beamSize;
            // qp.limit = static_cast<int>(qp.limit / (endpoints.second - endpoints.first));
 
            auto [pairElts, dist_cmps] = beam_search<Point, SubsetPointRange<T, Point, PointRange<T, Point>, uint32_t>, index_type>(query, G, naive_index.pr, 0, qp);

            auto frontier = pairElts.first;

            // filter for the points which are in the time range
            int found = 0;
            for (size_t i = 0; i < frontier.size() && found < k; i++) {
                if (naive_index.timestamps[frontier[i].first] >= min_time && naive_index.timestamps[frontier[i].first] <= max_time) {
                    out[found] = naive_index.pr.real_index(frontier[i].first);
                    dists[found] = frontier[i].second;
                    found++;
                }
            }
            return found;
        } else {
            // if this is the bottom level, query the naive index
            if (left == nullptr || right == nullptr) {
                return naive_index._range_knn(query, out, dists, left_end, right_end, min_time, max_time, k);
            }
            // if the desired range is split between the children, query both and then merge the results
            if (left_end < mid && right_end > mid) {
                auto left_out = parlay::sequence<index_type>::uninitialized(k);
                auto left_dists = parlay::sequence<float>::uninitialized(k);
                auto right_out = parlay::sequence<index_type>::uninitialized(k);
                auto right_dists = parlay::sequence<float>::uninitialized(k);

                size_t left_found, right_found;

                // query the left and right children in parallel
                parlay::par_do([&] () {
                    left_found = left->_range_knn(query, &left_out[0], &left_dists[0], left_end, mid, min_time, max_time, k);
                }, [&] () {
                    right_found = right->_range_knn(query, &right_out[0], &right_dists[0], 0, right_end - mid, min_time, max_time, k);
                });

                // merge the returned nearest neighbor lists (they're expected to be sorted by distance already)
                size_t i = 0, j = 0, found = 0;
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
                    size_t extra = std::min(right_found - j, k - found);
                    std::memcpy(&out[found], &right_out[j], extra);
                    std::memcpy(&dists[found], &right_dists[j], extra);
                    found += extra;
                }
                else if (j == right_found) {
                    size_t extra = std::min(left_found - i, k - found);
                    std::memcpy(&out[found], &left_out[i], extra);
                    std::memcpy(&dists[found], &left_dists[i], extra);
                    found += extra;
                }
                return found;
            }
            // if the range is entirely in the left child, just recurse into the left child
            else if (left_end < mid) {
                return left->_range_knn(query, out, dists, left_end, right_end, min_time, max_time, k);
            }
            // if the range is entirely in the right child, just recurse into the right child
            else {
                return right->_range_knn(query, out, dists, left_end - mid, right_end - mid, min_time, max_time, k);
            }
        }
    }

    // wrapper function for _range_knn
    // binary searches for the index endpoints based on the provided timestamp endpoints, then call _range_knn
    void range_knn(Point& query, index_type* out, std::pair<float, float> endpoints, size_t k) override {
        index_type start = 0;
        index_type end = naive_index.pr.size();

        index_type l = 0;
        index_type r = naive_index.pr.size();

        while (l < r) {
            index_type m = (l + r) / 2;
            if (naive_index.timestamps[m] < endpoints.first) l = m + 1;
            else r = m;
        }
        start = l;

        l = 0;
        r = naive_index.pr.size();

        while (l < r) {
            index_type m = (l + r) / 2;
            if (naive_index.timestamps[m] <= endpoints.second) l = m + 1;
            else r = m;
        }
        end = l;

        auto dists = parlay::sequence<float>::uninitialized(k);
        int found = _range_knn(query, out, &dists[0], start, end, endpoints.first, endpoints.second, k);
        if (found < k) {
            std::cout << "Warning: not enough points found" << std::endl;
        }
    }
 */
    /*parlay::sequence<std::pair<float, index_type>> _knn_with_real_index_distances(Point& query, size_t k) {
        auto [pairElts, dist_cmps] = beam_search<Point, SubsetPointRange<T, Point, PointRange<T, Point>, uint32_t>, index_type>(query, G, naive_index.pr, 0, default_query_params);

        parlay::sequence<std::pair<float, index_type>> frontier(k);

        for (size_t i = 0; i < k; i++) {
            frontier[i] = std::make_pair(pairElts.first[i].second, naive_index.pr.real_index(pairElts.first[i].first));
        }

        return frontier;
    }

    // endpoints here are local indices
    parlay::sequence<std::pair<float, index_type>> _index_range_knn_with_real_index_distances(Point& query, std::pair<index_type, index_type> endpoints, size_t k) {
        size_t size = endpoints.second - endpoints.first;

        if (size >= this->size()) {
            if (size > this->size()) {
                std::cout << "Warning: range larger than index size" << std::endl;
            }

            return _knn_with_real_index_distances(query, k);
        }

        if (size <= tiny_cutoff) {
            auto local_index_distances = naive_index._index_range_knn(query, k, std::make_pair(endpoints.first, endpoints.second - endpoints.first));

            parlay::sequence<std::pair<float, index_type>> result(k);

            for (size_t i = 0; i < k; i++) {
                result[i] = std::make_pair(local_index_distances[i].first, naive_index.pr.real_index(local_index_distances[i].second));
            }

            return result;
        } else if (size >= this->size() * overretrieval_cutoff) {
            // just do overretrieval
            auto [pairElts, dist_cmps] = beam_search<Point, SubsetPointRange<T, Point, PointRange<T, Point>, uint32_t>, index_type>(query, G, naive_index.pr, 0, default_overretrieval_params);

            auto frontier = pairElts.first;

            parlay::sequence<std::pair<float, index_type>> result(k);

            // filter + swap order of pair + convert to real index
            for (size_t i = 0; i < frontier.size(); i++) {
                if (frontier[i].first >= endpoints.first && frontier[i].first < endpoints.second) {
                    result.push_back(std::make_pair(frontier[i].second, naive_index.pr.real_index(frontier[i].first)));
                }
                if (result.size() == k) {
                    break;
                }
            }
            return result;
        } else {
            // we recurse on children
            if (left == nullptr || right == nullptr) {
                auto local_index_distances = naive_index._index_range_knn(query, k, std::make_pair(endpoints.first, endpoints.second - endpoints.first));

                parlay::sequence<std::pair<float, index_type>> result(k);

                for (size_t i = 0; i < k; i++) {
                    result[i] = std::make_pair(local_index_distances[i].first, naive_index.pr.real_index(local_index_distances[i].second));
                }

                return result;
            }

            // if range overlaps the left child
            if (endpoints.first < mid && endpoints.second > mid) {
                auto left_result = left->_index_range_knn_with_real_index_distances(query, std::make_pair(endpoints.first, mid), k);
                auto right_result = right->_index_range_knn_with_real_index_distances(query, std::make_pair(0, endpoints.second - mid), k);
                
                update_frontier(left_result, right_result, k);

                return left_result;
            } else if (endpoints.first < mid) {
                return left->_index_range_knn_with_real_index_distances(query, std::make_pair(endpoints.first, mid), k);
            } else {
                return right->_index_range_knn_with_real_index_distances(query, std::make_pair(endpoints.first - mid, endpoints.second - mid), k);
            }
        }
    }

    // void range_knn(Point& query, index_type* out, std::pair<float, float> endpoints, size_t k) override {
    //     auto endpoint_indices = naive_index._range_indices(endpoints);

    //     auto frontier = _index_range_knn_with_real_index_distances(query, endpoint_indices, k);

    //     for (size_t i = 0; i < k; i++) {
    //         out[i] = frontier[i].second;
    //     }
    // }

    // void range_knn(Point& query, index_type* out, std::pair<float, float> endpoints, size_t k) override {
    //     auto endpoint_indices = naive_index._range_indices(endpoints);

        if (endpoints.second - endpoints.first < exhaustive_fallback_cutoff) {
            naive_index.range_knn(query, out, endpoints, k);
        } else {
            ivf.range_knn(query, out, endpoints, k);
        }
    }*/

    /*void range_knn(Point& query, index_type* out, std::pair<float, float> endpoints, size_t k) override {
        naive_index.range_knn(query, out, endpoints, k);
    }*/

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
    }

    void overretrieval_range_knn(Point& query, index_type* out, std::pair<float, float> endpoints, size_t k) {
        QueryParams qp = default_query_params;
        qp.k = qp.beamSize;
        // qp.limit = static_cast<int>(qp.limit / (endpoints.second - endpoints.first));

        auto [pairElts, dist_cmps] = beam_search<Point, SubsetPointRange<T, Point, PointRange<T, Point>, uint32_t>, index_type>(query, G, naive_index.pr, 0, qp);

        auto frontier = pairElts.first;

        // filter for the points which are in the time range
        int found = 0;
        for (size_t i = 0; i < frontier.size() && found < k; i++) {
            if (naive_index.timestamps[frontier[i].first] >= endpoints.first && naive_index.timestamps[frontier[i].first] <= endpoints.second) {
                out[found] = naive_index.pr.real_index(frontier[i].first);
                found++;
            }
        }
    }

    size_t size() const override {
        return naive_index.pr.size();
    }

    size_t dims() const override {
        return naive_index.pr.dims;
    }

    size_t aligned_dims() const override {
        return naive_index.pr.aligned_dims;
    }
};

