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

#define DIM 100 // dimensionality of the data
#define ALIGNED_DIM 112 
#define K 100 // number of neighbors to return

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
        std::cout << "Built small indices in " << t.next_time() << " seconds" << std::endl;

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

        std::cout << "Sorted queries in " << t.next_time() << " seconds" << std::endl;

        #ifdef PROD
        // this code is faster than the more informative code below
        // does not consider aligned dim, probably wrong
            parlay::parallel_for(0, num_queries, [&](index_type i) {
                auto [query_type, category, start, end, index] = queries[i];
                Point query = Point(query_vectors + i * DIM, DIM);
                switch (query_type) {
                    case 0:
                        big_index.knn(query, out + i * K, K);
                        break;
                    case 1:
                        categorical_indices[category].knn(query, out + i * K, K);
                        break;
                    case 2:
                        big_index.range_knn(query, out + i * K, std::make_pair(start, end), K);
                        break;
                    case 3:
                        categorical_indices[category].range_knn(query, out + i * K, std::make_pair(start, end), K);
                        break;
                    default:
                        throw std::runtime_error("Invalid query type");
                }
            });
        #endif

        #ifndef PROD
            // interpretable query code that is probably a wee bit slower end to end
            // order should probably be big->range->categorical->categorical range
            // or even better big->range->categorical+categorical range

            // run big queries
            parlay::parallel_for(0, query_type_count[0], [&](index_type i) {
                auto [query_type, category, start, end, index] = queries[i];
                Point query = Point(query_vectors + i * ALIGNED_DIM, DIM, ALIGNED_DIM, index);
                big_index.knn(query, out + index * K, K);
            });

            double big_time = t.next_time();
            std::cout << "Ran " << query_type_count[0] << " big queries in " << big_time << " seconds (QPS: " << query_type_count[0] / big_time << ")" << std::endl;

            // run range queries
            parlay::parallel_for(query_type_count[0] + query_type_count[1], query_type_count[0] + query_type_count[1] + query_type_count[2], [&](index_type i) {
                auto [query_type, category, start, end, index] = queries[i];
                Point query = Point(query_vectors + i * ALIGNED_DIM, DIM, ALIGNED_DIM, index);
                big_index.range_knn(query, out + index * K, std::make_pair(start, end), K);
            });

            double range_time = t.next_time();
            std::cout << "Ran " << query_type_count[2] << " range queries in " << range_time << " seconds (QPS: " << query_type_count[2] / range_time << ")" << std::endl;

            // run categorical queries
            parlay::parallel_for(query_type_count[0], query_type_count[0] + query_type_count[1], [&](index_type i) {
                auto [query_type, category, start, end, index] = queries[i];
                Point query = Point(query_vectors + i * ALIGNED_DIM, DIM, ALIGNED_DIM, index);
                categorical_indices[category]->knn(query, out + index * K, K);
            });

            double categorical_time = t.next_time();
            std::cout << "Ran " << query_type_count[1] << " categorical queries in " << categorical_time << " seconds (QPS: " << query_type_count[1] / categorical_time << ")" << std::endl;

            // run categorical range queries
            parlay::parallel_for(query_type_count[0] + query_type_count[1] + query_type_count[2], num_queries, [&](index_type i) {
                auto [query_type, category, start, end, index] = queries[i];
                Point query = Point(query_vectors + i * ALIGNED_DIM, DIM, ALIGNED_DIM, index);
                categorical_indices[category]->range_knn(query, out + index * K, std::make_pair(start, end), K);
            });

            double categorical_range_time = t.next_time();
            std::cout << "Ran " << query_type_count[3] << " categorical range queries in " << categorical_range_time << " seconds (QPS: " << query_type_count[3] / categorical_range_time << ")" << std::endl;

        #endif

        std::cout << "Ran all queries in " << t.next_time() << " seconds" << std::endl;
    }

private:
	void init_categorical_indices() {
		auto vectors_by_label = parlay::sequence<parlay::sequence<uint32_t>>(max_label + 1);
		auto timestamps_by_label = parlay::sequence<parlay::sequence<float>>(max_label + 1);

		for (int i = 0; i < points.size(); i++) {
			vectors_by_label[labels[i]].push_back(i);
			timestamps_by_label[labels[i]].push_back(timestamps[i]);
		}

		categorical_indices = parlay::sequence<std::unique_ptr<VirtualIndex<T, Point>>>::uninitialized(max_label + 1);

		// Profile if SmallIndex init being parallelized will also help, since there are a few labels with a large number of points.
        for (int i = 0; i <= categorical_indices.size(); i++) {
		// parlay::parallel_for(0, categorical_indices.size(), [&] (size_t i) {
			if (vectors_by_label[i].size() == 0) return;
			categorical_indices[i] = std::make_unique<SmallIndex>();
			categorical_indices[i]->fit(points, timestamps_by_label[i], vectors_by_label[i]);
		// }); 
        }
	}
};
