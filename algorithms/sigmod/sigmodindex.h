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

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <filesystem>
#include <fstream>
#include <utility>
#include <variant>
#include <vector>


#define DIM 100 // dimensionality of the data

using index_type = uint32_t;
using T = float;
using Point = Euclidian_Point<T>;




/* The index itself, supporting construction from the competition format, querying in the competition format, and the underlying types of query that entails 

BigIndex: The index we build over all the points, and probably the large categorical filters. This is probably a Vamana graph.
SmallIndex: The index we build over the small categorical filters, which is probably internally a set of points we do exhaustive search on which may be reordered.
RangeIndex: The index we build for range filters.

Each index CAN own its own copy of the points, but the constructor will provide a reference to the full dataset (which an index can hold onto and trust remains valid for the lifetime of the SigmodIndex), the relevant points, and the corresponding relevant metadata.

All methods of member indices which return point indices should return them relative to the original dataset. The member index is responsible for holding onto its own points' true indices.

The component indices are templatized for easy comparison, and supposing we only instantiate them once, they shouldn't cause compile time overhead as far as I know.
*/
template <typename BigIndex, typename SmallIndex, typename RangeIndex>
class SigmodIndex {
    PointRange<T, Point> points;
    parlay::sequence<index_type> labels;
    parlay::sequence<float> timestamps;

    BigIndex big_index;
    
    RangeIndex range_index;

    public:

    /* probably want to do something real here, but not real init */
    SigmodIndex() = default;

    /* Construct the index from the competition format */
    void build_index(const std::string& filename) {
        throw std::runtime_error("Not implemented");
    }

    /* query the index with the competition format */
    void competition_query(const std::string& filename, index_type* out) {
        throw std::runtime_error("Not implemented");
    }


};