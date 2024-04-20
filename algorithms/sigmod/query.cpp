/* Builds and queries an index for provided data and query files */

#include <iostream>
#include <fstream>

#include "sigmodindex.h"

#define K 100

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <data_file> <query_file> <out_file>" << std::endl;
        return 1;
    }

    SigmodIndex<VamanaIndex<float, Euclidian_Point<float>>, NaiveIndex<float, Euclidian_Point<float>>> index;

    index.build_index(argv[1]);

    // need length of queries to get the right size of the output array
    std::ifstream file(argv[2]);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file " + std::string(argv[2]));
    }

    uint32_t num_queries;
    file.read((char*)&num_queries, sizeof(uint32_t));
    file.close();

    uint32_t *out = new uint32_t[num_queries * K];

    index.competition_query(argv[2], out);

    std::ofstream out_file(argv[3]);
    if (!out_file.is_open()) {
        throw std::runtime_error("Unable to open file " + std::string(argv[3]));
    }

    out_file.write((char*)out, num_queries * K * sizeof(uint32_t));
    out_file.close();

    return 0;
}