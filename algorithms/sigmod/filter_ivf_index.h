/* An ivf index for filtered queries that checks a target number of matching points exhaustively */

#pragma once

#include "small_index.h"

#include "../utils/point_range.h"
#include "../utils/graph.h"
#include "../utils/beamSearch.h"
#include "../utils/fp32_sq_euclidean.h"
#include "../vamana/index.h"
#include "../IVF/clustering.h"

#include <algorithm>
#include <iostream>
#include <cstring>

using index_type = uint32_t;

#define MAX_ITERS 20
#define SUBSAMPLE 10

#define MIN_MATCHES 1000

template<typename T, typename Point>
struct Filter_IVF {
    // SubsetPointRange<T, Point, PointRange<T, Point>, uint32_t> *pr = nullptr;
    parlay::sequence<NaiveIndex<T, Point>> clusters; // each cluster has a naive index for the points in that cluster
    std::unique_ptr<T[]> centroid_data; // the centroids of the clusters, which needn't really be in a point range

    // should add something to store the range of the timestamps for each cluster

    size_t dim, aligned_dim, n;

    size_t max_iters = MAX_ITERS;
    size_t subsample = SUBSAMPLE;

    Filter_IVF() = default;

    Filter_IVF(SubsetPointRange<T, Point, PointRange<T, Point>, index_type> points, parlay::sequence<float> timestamps, size_t n_clusters) : dim(points.dims), aligned_dim(points.aligned_dims), n(points.size()), clusters(n_clusters) {
        KMeansClusterer<T, Point, index_type> clusterer(n_clusters);

        clusterer.subsample = subsample;
        clusterer.max_iters = max_iter;

        auto cluster_indices = clusterer.cluster(*(points.pr), parlay::tabulate(points.size(), [](index_type i) { return i; }));
        // this will break silently and heinously if pr is not a copy, hence this assert
        assert(points.pr->size() == points.size());

        // we need to map these indices back to the subset so they can later be mapped back again to the original indices
        // or maybe not because we want to pass actual indices to the NaiveIndex
        // for (size_t i = 0; i < cluster_indices.size(); i++) {
        //     for (size_t j = 0; j < cluster_indices[i].size(); j++) {
        //         cluster_indices[i][j] = points.real_to_subset[cluster_indices[i][j]];
        //     }
        // }

        this->centroid_data = std::make_unique<T[]>(n_clusters * aligned_dim);

        parlay::parallel_for(0, n_clusters, [&] (size_t i) {
            clusters[i].fit(*(points.pr), timestamps, cluster_indices[i]);
            // compute centroid
            for (size_t j = 0; j < dim; j++) {
                double sum = 0;
                for (size_t k = 0; k < cluster_indices[i].size(); k++) {
                    sum += (*(points.pr))[cluster_indices[i][k]].values[j];
                }
                this->centroid_data[i * aligned_dim + j] = static_cast<T>(sum / cluster_indices[i].size());
            }
        });
    }

    /* This will not be robust to changes in T or Point or dim. very brittle really. */
    parlay::sequence<std::pair<float, index_type>> centroid_comparison(Point query) {
        parlay::sequence<std::pair<float, index_type>> results(clusters.size());

        for (size_t i = 0; i < clusters.size(); i++) {
            results[i] = std::make_pair(sq_euclidean<100>(query.values, this->centroid_data.get() + i * aligned_dim), i);
        }

        std::sort(results.begin(), results.end());

        return results;
    }

    /* updates the provided results sequence with points in the relevant cluster */
    void cluster_range_knn(Point query, size_t k, std::pair<float, float> range, parlay::sequence<std::pair<float, index_type>>& results, size_t* matches, size_t cluster) {
        // requires kind of stupid monkeying with internal state of NaiveIndex
        auto range_indices = clusters[cluster]._range_indices(range);

        size_t start = range_indices.first;
        size_t length = range_indices.second - range_indices.first;

        *matches += length;

        auto cluster_results = clusters[cluster]._index_range_knn(query, k, std::make_pair(start, length));

        // note that these distances are already sorted
        for (auto& candidate : cluster_results) {
            if (results.size() < k) {
                results.push_back(std::make_pair(candidate.first, clusters[cluster].pr.real_index(candidate.second)));
                std::sort(results.begin(), results.end());
            } else if (candidate.first < results.back().first) {
                results.back() = std::make_pair(candidate.first, clusters[cluster].pr.real_index(candidate.second));
            } else if (candidate.first >= results.back().first) {
                break;
            }
        }
    }

    void range_knn(Point query, index_type* out, std::pair<float, float> endpoints, size_t k) {
        auto centroid_results = centroid_comparison(query);

        size_t matches = 0;

        parlay::sequence<std::pair<float, index_type>> results;

        for (size_t i = 0; i < centroid_results.size(); i++) {
            if (matches >= MIN_MATCHES) {
                break;
            }
            if (i > centroid_results.size() / 2 && centroid_results.size() > 10) {
                std::cerr << "Warning: more than half of the clusters have been checked for a query with range length " << endpoints.second - endpoints.first << std::endl;
            }

            cluster_range_knn(query, k, endpoints, results, &matches, centroid_results[i].second);
        }

        // std::sort(results.begin(), results.end());

        for (size_t i = 0; i < k; i++) {
            out[i] = results[i].second;
        }
    }
};
