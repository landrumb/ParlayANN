/* The main index for the 2024 SIGMOD student competition */

#pragma once

#include "parlay/internal/get_time.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"
#include "parlay/delayed_sequence.h"
#include "parlay/slice.h"
#include "parlay/internal/group_by.h"

#include "../utils/point_range.h"
#include "../utils/types.h"
#include "../utils/euclidian_point.h"

#include <cstdint>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <filesystem>
#include <fstream>
#include <utility>
#include <variant>
#include <vector>
#include <cassert>

#include "virtual_index.h"
#include "small_index.h"
#include "big_index.h"

//#define EXACT

#define DIM 100 // dimensionality of the data
#define ALIGNED_DIM 112 
#define K 100 // number of neighbors to return

#define DEFAULT_CUTOFF 50'000

using index_type = uint32_t;
using T = float;
using Point = Euclidian_Point<T>;

// type, category, start, end, index
using Query = std::tuple<int, int, float, float, index_type>;


/* The index itself, supporting construction from the competition format, querying in the competition format, and the underlying types of query that entails 

BigIndex: The index we build over all the points, and probably the large categorical filters. This is probably a Vamana graph.
SmallIndex: The index we build over the small categorical filters, which is probably internally a set of points we do exhaustive search on which may be reordered.
RangeIndex: The index we build for range filters.

Each index CAN own its own copy of the points, but the constructor will provide a reference to the full dataset (which an index can hold onto and trust remains valid for the lifetime of the SigmodIndex), the relevant points, and the corresponding relevant metadata.

All methods of member indices which return point indices should return them relative to the original dataset. The member index is responsible for holding onto its own points' true indices.

The component indices are templatized for easy comparison, and supposing we only instantiate them once, they shouldn't cause compile time overhead as far as I know.
*/

template <typename BigIndex, typename SmallIndex>
class SigmodIndex {
public:
    PointRange<T, Point> points;
    parlay::sequence<index_type> labels; // the label for each point
    int max_label;
    parlay::sequence<float> timestamps; // the timestamp for each point

    BigIndex big_index;
    parlay::sequence<std::unique_ptr<VirtualIndex<T, Point>>> categorical_indices;
    parlay::sequence<parlay::sequence<std::unique_ptr<BigIndex>>> range_indices;

    double qps_per_case[4] = {0., 0., 0., 0.};

    size_t cutoff = DEFAULT_CUTOFF;

    /* probably want to do something real here, but not real init */
    SigmodIndex() {};

    void load_points(const std::string& filename) {
        std::ifstream reader(filename);
        if (!reader.is_open()) {
            throw std::runtime_error("Unable to open file " + filename);
        }

        uint32_t num_points;
        reader.read((char*)&num_points, 4);
        T *values = new T[num_points * DIM];
        labels = parlay::sequence<index_type>::uninitialized(num_points);
        timestamps = parlay::sequence<float>::uninitialized(num_points);

        max_label = 0;
        for (int i = 0; i < num_points; i++) {
            float temp;
            reader.read((char*)&temp, 4);
            labels[i] = (uint32_t)temp;
            if (labels[i] > max_label) max_label = labels[i];
            reader.read((char*)&timestamps[i], 4);
            reader.read((char*)&values[DIM * i], DIM * sizeof(T));
        }

        points = PointRange<T, Point>(values, num_points, DIM);

        delete[] values;
        reader.close();
    }

    /* Construct the index from the competition format */
    void build_index(const std::string& filename) {
        parlay::internal::timer t;
        t.start();

        load_points(filename);
        std::cout << "Read points in " << t.next_time() << " seconds" << std::endl;

        init_categorical_indices();
        std::cout << "Built categorical indices in " << t.next_time() << " seconds" << std::endl;
        
        init_range_indices();
        std::cout << "Built range indices in " << t.next_time() << " seconds" << std::endl;

        big_index.fit(points, timestamps);
        std::cout << "Built big index in " << t.next_time() << " seconds" << std::endl;
    }

    /* query the index with the competition format 
    
    I think if it's worth it (possible) we can read in parallel
    */
    void competition_query(const std::string& filename, index_type* out) {
        parlay::internal::timer t;
        t.start();

        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Unable to open file " + filename);
        }

        uint32_t num_queries;
        file.read((char*)&num_queries, sizeof(uint32_t));

        // where we pack the queries in the order we read them
        T* query_vectors = static_cast<T*>(std::aligned_alloc(64, num_queries * ALIGNED_DIM * sizeof(T)));

        Query* queries = new Query[num_queries];
        
        int query_type_count[4] = {0, 0, 0, 0}; // will be annoying if parallelized

        for (index_type i = 0; i < num_queries; i++) {
            float query_type, category, start, end;

            file.read((char*)&query_type, sizeof(float));
            file.read((char*)&category, sizeof(float));
            file.read((char*)&start, sizeof(float));
            file.read((char*)&end, sizeof(float));

            query_type_count[static_cast<int>(query_type)]++;

            queries[i] = Query(static_cast<int>(query_type), static_cast<int>(category), start, end, i);

            // read in the query vector
            file.read((char*)&query_vectors[i *ALIGNED_DIM], DIM * sizeof(T));
        }

        file.close();

        std::cout << "Read queries in " << t.next_time() << " seconds" << std::endl;

        // this should be lexocographic sort over the strategically ordered Query tuple
        // should be parallel but am putting off remembering how slice initialization works
        std::sort(queries, queries + num_queries);

        // important to note that we do NOT sort the query vectors, and they should be accessed by index

        std::cout << "Sorted queries in " << t.next_time() << " seconds" << std::endl;

#ifndef EXACT
#ifdef PROD
        // this code is faster than the more informative code below
        // does not consider aligned dim, probably wrong
        parlay::parallel_for(0, num_queries, [&](index_type i) {
            auto [query_type, category, start, end, index] = queries[i];
            Point query = Point(query_vectors + i * ALIGNED_DIM, DIM, ALIGNED_DIM, index);
            switch (query_type) {
                case 0:
                    big_index.knn(query, out + i * K, K);
                    break;
                case 1:
                    categorical_indices[category]->knn(query, out + i * K, K);
                    break;
                case 2:
                    big_index.range_knn(query, out + i * K, std::make_pair(start, end), K);
                    break;
                case 3:
                    categorical_indices[category]->range_knn(query, out + i * K, std::make_pair(start, end), K);
                    break;
                default:
                    throw std::runtime_error("Invalid query type");
            }
        });

#else
        
        // interpretable query code that is probably a wee bit slower end to end
        // order should probably be big->range->categorical->categorical range
        // or even better big->range->categorical+categorical range

        // run big queries
        big_index_batch_query(queries, query_vectors, out, query_type_count[0]);

        double big_time = t.next_time();
        qps_per_case[0] = query_type_count[0] / big_time;
        std::cout << "Ran " << query_type_count[0] << " big queries in " << big_time << " seconds (QPS: " << query_type_count[0] / big_time << ")" << std::endl;

        // run range queries
        //big_index_range_query(queries + query_type_count[0] + query_type_count[1], query_vectors, out, query_type_count[2]);
        windowed_range_queries(queries, query_type_count[0] + query_type_count[1], query_type_count[0] + query_type_count[1] + query_type_count[2], query_vectors, out);

        double range_time = t.next_time();
        qps_per_case[2] = query_type_count[2] / range_time;
        std::cout << "Ran " << query_type_count[2] << " range queries in " << range_time << " seconds (QPS: " << query_type_count[2] / range_time << ")" << std::endl;

        // run categorical queries
        categorical_query(queries + query_type_count[0], query_vectors, out, query_type_count[1]);

        double categorical_time = t.next_time();
        qps_per_case[1] = query_type_count[1] / categorical_time;
        std::cout << "Ran " << query_type_count[1] << " categorical queries in " << categorical_time << " seconds (QPS: " << query_type_count[1] / categorical_time << ")" << std::endl;

        // run categorical range queries
        categorical_range_query(queries + query_type_count[0] + query_type_count[1] + query_type_count[2], query_vectors, out, query_type_count[3]);

        double categorical_range_time = t.next_time();
        qps_per_case[3] = query_type_count[3] / categorical_range_time;
        std::cout << "Ran " << query_type_count[3] << " categorical range queries in " << categorical_range_time << " seconds (QPS: " << query_type_count[3] / categorical_range_time << ")" << std::endl;

#endif
#else

        // exact querying for ground truth

        // run big queries
        exact_query(queries, query_vectors, out, query_type_count[0]);

        double big_time = t.next_time();
        qps_per_case[0] = query_type_count[0] / big_time;
        std::cout << "Ran " << query_type_count[0] << " big queries in " << big_time << " seconds (QPS: " << query_type_count[0] / big_time << ")" << std::endl;

        // run range queries
        exact_range_query(queries + query_type_count[0] + query_type_count[1], query_vectors, out, query_type_count[2]);

        double range_time = t.next_time();
        qps_per_case[2] = query_type_count[2] / range_time;
        std::cout << "Ran " << query_type_count[2] << " range queries in " << range_time << " seconds (QPS: " << query_type_count[2] / range_time << ")" << std::endl;

        // run categorical queries
        exact_categorical_query(queries + query_type_count[0], query_vectors, out, query_type_count[1]);

        double categorical_time = t.next_time();
        qps_per_case[1] = query_type_count[1] / categorical_time;
        std::cout << "Ran " << query_type_count[1] << " categorical queries in " << categorical_time << " seconds (QPS: " << query_type_count[1] / categorical_time << ")" << std::endl;

        // run categorical range queries
        exact_categorical_range_query(queries + query_type_count[0] + query_type_count[1] + query_type_count[2], query_vectors, out, query_type_count[3]);

        double categorical_range_time = t.next_time();
        qps_per_case[3] = query_type_count[3] / categorical_range_time;
        std::cout << "Ran " << query_type_count[3] << " categorical range queries in " << categorical_range_time << " seconds (QPS: " << query_type_count[3] / categorical_range_time << ")" << std::endl;

#endif

        std::cout << "Ran all queries in " << t.total_time() << " seconds" << std::endl;
    }

private:
    void init_categorical_indices() {
        auto vectors_by_label = parlay::sequence<parlay::sequence<index_type>>(max_label + 1);
        auto timestamps_by_label = parlay::sequence<parlay::sequence<T>>(max_label + 1);

        for (int i = 0; i < points.size(); i++) {
            vectors_by_label[labels[i]].push_back(i);
            timestamps_by_label[labels[i]].push_back(timestamps[i]);
        }

        categorical_indices = parlay::sequence<std::unique_ptr<VirtualIndex<T, Point>>>::from_function(max_label + 1, [&] (size_t i) {
#ifdef EXACT
            std::unique_ptr<VirtualIndex<T, Point>> ptr = std::make_unique<SmallIndex>();
            ptr->fit(points, timestamps_by_label[i], vectors_by_label[i]);
            return ptr;
#else
            if (vectors_by_label[i].size() > cutoff) {
                std::unique_ptr<VirtualIndex<T, Point>> ptr = std::make_unique<BigIndex>();
                ptr->fit(points, timestamps_by_label[i], vectors_by_label[i]);
                return ptr;
            } else {
                std::unique_ptr<VirtualIndex<T, Point>> ptr = std::make_unique<SmallIndex>();
                ptr->fit(points, timestamps_by_label[i], vectors_by_label[i]);
                return ptr;
            }
#endif
        });
    }

    void init_range_indices() {
#ifndef EXACT
        auto sorted_index_map = parlay::sequence<index_type>::from_function(points.size(), [] (size_t i) -> index_type { return i; });
        parlay::sort_inplace(sorted_index_map, [&] (index_type i, index_type j) -> bool {
            return timestamps[i] < timestamps[j];
        });

        range_indices = parlay::sequence<parlay::sequence<std::unique_ptr<BigIndex>>>();
        for (int level_scale = 1; points.size() / level_scale > DEFAULT_CUTOFF; level_scale *= 2) {
            std::cout << "Building windows of size 1/" << level_scale << "..." << std::flush;
            range_indices.push_back(parlay::sequence<std::unique_ptr<BigIndex>>::from_function(2 * level_scale - 1, [&] (size_t i) {
                size_t range_start = i * points.size() / (2 * level_scale);
                size_t range_end = (i + 2) * points.size() / (2 * level_scale);
                auto vectors_by_range = parlay::sequence<index_type>::from_function(range_end - range_start,
                    [&] (size_t i) -> index_type {
                        return sorted_index_map[range_start + i];
                    });
                auto timestamps_by_range = parlay::sequence<T>::from_function(vectors_by_range.size(),
                    [&] (size_t i) -> T {
                        return timestamps[vectors_by_range[i]];
                    });

                std::unique_ptr<BigIndex> ptr = std::make_unique<BigIndex>();
                ptr->fit(points, timestamps_by_range, vectors_by_range);
                return ptr;
            }));
            std::cout << " Done" << std::endl;
        }
#endif
    }
    /* 
    returns the index of the window a range with a given start will resolve to
    within the specified level
     */
    inline unsigned int which_window_in_level(int level, float start) {
        int window = static_cast<int>((1 << (level + 1)) * start);
        if (window == (1 << (level + 1)) - 1) window--;
        return window;
    }
    /* 
    returns the index of the index of a window given its level and index within that level
    */
    inline unsigned int encode_window(int level, int window_in_level) {
        return (1 << (level + 1)) - 1 - level + window_in_level;
    }

    template <typename Range>
    void windowed_range_queries(Range&& range_queries, size_t queries_start, size_t queries_end, T *query_vectors, index_type *out) {
        auto bucketed_queries = parlay::sequence<std::pair<unsigned int, Query>>::from_function(queries_end - queries_start,
            [&] (size_t i) {
                auto [query_type, category, start, end, index] = range_queries[queries_start + i];

                int level = end - start >= 3.0 / 8.0 ? 0 : static_cast<int>(floor(log2(3.0 / 8.0 / (end - start)))) + 1;

                if (level >= range_indices.size()) return std::make_pair(0u, range_queries[queries_start + i]);
                int window_in_level = which_window_in_level(level, start);
                return std::make_pair(encode_window(level, window_in_level), range_queries[queries_start + i]);
            });

        auto grouped_queries = parlay::group_by_index(bucketed_queries, encode_window(range_indices.size(), 0));
        /*for (int i = 0; i < grouped_queries.size(); i++) {
            std::cout << "Group " << i << " contains " << grouped_queries[i].size() << " range queries" << std::endl;
        }*/

        double time;
        parlay::internal::timer timer;
        std::cout << "Starting range queries" << std::endl;
        timer.start();

        parlay::parallel_for(0, grouped_queries[0].size(), [&] (size_t i) {
                auto [query_type, category, start, end, index] = grouped_queries[0][i];
                Point query = Point(query_vectors + index * ALIGNED_DIM, DIM, ALIGNED_DIM, index);
                range_indices[0][0]->naive_index.range_knn(query, out + index * K, std::make_pair(start, end), K);
            });
        time = timer.next_time();
        std::cout << "Exhaustive queries: " << grouped_queries[0].size() << " queries in " << time << " seconds (QPS: " << grouped_queries[0].size() / time << ")" << std::endl;
        int group_id = 1;
        int level_id = 0;
        for (auto& window_level : range_indices) {
            parlay::parallel_for(0, window_level.size(), [&] (size_t i) {
                    parlay::parallel_for(0, grouped_queries[group_id + i].size(), [&] (size_t j) {
                            auto [query_type, category, start, end, index] = grouped_queries[group_id + i][j];
                            Point query = Point(query_vectors + index * ALIGNED_DIM, DIM, ALIGNED_DIM, index);
                            window_level[i]->range_knn(query, out + index * K, std::make_pair(start, end), K);
                        }, 1);
                });

            time = timer.next_time();
            int level_size = 0;
            for (int i = 0; i < window_level.size(); i++) level_size += grouped_queries[group_id + i].size();
            std::cout << "Level " << ++level_id << ": " << level_size << " queries in " << time << " seconds (QPS: " << level_size / time << ")" << std::endl;

            group_id += window_level.size();
        }
/*
        int group_id = 1;
        int level_id = 0;
        auto& window_level = range_indices[2];
            parlay::parallel_for(0, window_level.size(), [&] (size_t i) {
                    parlay::parallel_for(0, grouped_queries[group_id + i].size(), [&] (size_t j) {
                            auto [query_type, category, start, end, index] = grouped_queries[group_id + i][j];
                            Point query = Point(query_vectors + index * ALIGNED_DIM, DIM, ALIGNED_DIM, index);
                            window_level[i]->range_knn(query, out + index * K, std::make_pair(start, end), K);
                        }, 1);
                });

            time = timer.next_time();
            int level_size = 0;
            for (int i = 0; i < window_level.size(); i++) level_size += grouped_queries[group_id + i].size();
            std::cout << "Level " << ++level_id << ": " << level_size << " queries in " << time << " seconds (QPS: " << level_size / time << ")" << std::endl;

            group_id += window_level.size();*/
    }

    void execute_range_queries() {}
        
    inline void big_index_batch_query(Query* queries, T* query_vectors, index_type* out, size_t big_query_count) {
        parlay::parallel_for(0, big_query_count, [&](index_type i) {
            auto [query_type, category, start, end, index] = queries[i];
            Point query = Point(query_vectors + i * ALIGNED_DIM, DIM, ALIGNED_DIM, index);
            big_index.knn(query, out + index * K, K);
        });
        // move all the big query vectors into a contiguous block
        /*T* big_query_vectors = static_cast<T*>(std::aligned_alloc(64, big_query_count * ALIGNED_DIM * sizeof(T)));

        // should be parallel perhaps
        for (index_type i = 0; i < big_query_count; i++) {
            std::memcpy(big_query_vectors + i * ALIGNED_DIM, query_vectors + std::get<4>(queries[i]) * ALIGNED_DIM, DIM * sizeof(T));
        }

        index_type** out_ptrs = static_cast<index_type**>(malloc(big_query_count * sizeof(index_type*)));

        for (index_type i = 0; i < big_query_count; i++) {
            out_ptrs[i] = out + std::get<4>(queries[i]) * K;
        }

        // process queries directly for debug
        parlay::parallel_for(0, big_query_count, [&](index_type i) {
            auto [query_type, category, start, end, index] = queries[i];
            Point query = Point(big_query_vectors + i * ALIGNED_DIM, DIM, ALIGNED_DIM, index);
            big_index.knn(query, out_ptrs[i], K);
        });

        // big_index.batch_knn(big_query_vectors, out_ptrs, K, big_query_count, true);

        free(big_query_vectors);
        free(out_ptrs);*/
    }

    inline void big_index_range_query(Query* queries, T* query_vectors, index_type* out, size_t range_query_count) {
        parlay::parallel_for(0, range_query_count, [&](index_type i) {
                auto [query_type, category, start, end, index] = queries[i];
                Point query = Point(query_vectors + index * ALIGNED_DIM, DIM, ALIGNED_DIM, index);
                big_index.range_knn(query, out + index * K, std::make_pair(start, end), K);
            });
    }

    inline void categorical_query(Query* queries, T* query_vectors, index_type* out, size_t categorical_query_count) {
        // want to group queries and run them with batch_knn

        // prefix sum the number of queries for each label
        parlay::sequence<index_type> label_counts = parlay::sequence<index_type>(max_label + 2);
        label_counts[0] = 0;
        for (int i = 0; i < categorical_query_count; i++) {
            label_counts[std::get<1>(queries[i]) + 1]++; // +1 because we want the prefix sum to have 0 at the beginning
            // this makes label_counts[i-1] the start of that label's queries
            // and label_counts[i] the end (exclusive) of that label's queries
        }
        for (int i = 1; i <= max_label + 1; i++) {
            label_counts[i] += label_counts[i - 1];
        }

        // print first 3 elements for sanity
        std::cout << "label_counts: " << label_counts[0] << " " << label_counts[1] << " " << label_counts[2] << std::endl;

        T* categorical_query_vectors = static_cast<T*>(std::aligned_alloc(64, categorical_query_count * ALIGNED_DIM * sizeof(T)));
        index_type** out_ptrs = static_cast<index_type**>(malloc(categorical_query_count * sizeof(index_type*)));

        parlay::parallel_for(0, categorical_query_count, [&](index_type i) {
            auto [query_type, category, start, end, index] = queries[i];
            std::memcpy(categorical_query_vectors + i * ALIGNED_DIM, query_vectors + index * ALIGNED_DIM, DIM * sizeof(T));
            out_ptrs[i] = out + index * K;
        });

        parlay::parallel_for(1, max_label + 1, [&](index_type c) {
            if (label_counts[c - 1] != label_counts[c]) {
                categorical_indices[c]->batch_knn(categorical_query_vectors + label_counts[c - 1] * ALIGNED_DIM, out_ptrs + label_counts[c - 1], K, label_counts[c] - label_counts[c - 1], (label_counts[c] - label_counts[c - 1] > 50));
            }
        });
    }

    inline void categorical_range_query(Query* queries, T* query_vectors, index_type* out, size_t categorical_range_query_count) {
        parlay::parallel_for(0, categorical_range_query_count, [&](index_type i) {
            auto [query_type, category, start, end, index] = queries[i];
            Point query = Point(query_vectors + index * ALIGNED_DIM, DIM, ALIGNED_DIM, index);
            categorical_indices[category]->range_knn(query, out + index * K, std::make_pair(start, end), K);
        });
    }
        
    inline void exact_query(Query* queries, T* query_vectors, index_type* out, size_t big_query_count) {
        parlay::parallel_for(0, big_query_count, [&](index_type i) {
            auto [query_type, category, start, end, index] = queries[i];
            Point query = Point(query_vectors + i * ALIGNED_DIM, DIM, ALIGNED_DIM, index);
            big_index.naive_index.knn(query, out + index * K, K);
        });
    }

    inline void exact_range_query(Query* queries, T* query_vectors, index_type* out, size_t range_query_count) {
        parlay::parallel_for(0, range_query_count, [&](index_type i) {
                auto [query_type, category, start, end, index] = queries[i];
                Point query = Point(query_vectors + index * ALIGNED_DIM, DIM, ALIGNED_DIM, index);
                big_index.naive_index.range_knn(query, out + index * K, std::make_pair(start, end), K);
            });
    }

    inline void exact_categorical_query(Query* queries, T* query_vectors, index_type* out, size_t categorical_query_count) {
        parlay::parallel_for(0, categorical_query_count, [&](index_type i) {
            auto [query_type, category, start, end, index] = queries[i];
            Point query = Point(query_vectors + index * ALIGNED_DIM, DIM, ALIGNED_DIM, index);
            categorical_indices[category]->knn(query, out + index * K, K);
        });
    }

    inline void exact_categorical_range_query(Query* queries, T* query_vectors, index_type* out, size_t categorical_range_query_count) {
        parlay::parallel_for(0, categorical_range_query_count, [&](index_type i) {
            auto [query_type, category, start, end, index] = queries[i];
            Point query = Point(query_vectors + index * ALIGNED_DIM, DIM, ALIGNED_DIM, index);
            categorical_indices[category]->range_knn(query, out + index * K, std::make_pair(start, end), K);
        });
    }
};

