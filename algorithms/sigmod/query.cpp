/* Builds and queries an index for provided data and query files */
#include "parlay/internal/get_time.h"

#include <iostream>
#include <fstream>

#include "sigmodindex.h"

#define K 100

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <optional CSV flag -r> <data_file> <query_file> <out_file>" << std::endl;
        return 1;
    }

    int arg_offset = 0;
    if (strcmp(argv[1], "-r") == 0) {
        if (argc < 5) {
            std::cerr << "Usage: " << argv[0] << " <optional CSV flag -r> <data_file> <query_file> <out_file>" << std::endl;
            return 1;
        }
        arg_offset = 1;
    }

    parlay::internal::timer t;
    t.start();

    SigmodIndex<VamanaIndex<float, Euclidian_Point<float>>, NaiveIndex<float, Euclidian_Point<float>>> index;

    index.build_index(argv[1 + arg_offset]);

    double build_time = t.next_time();

    std::cout << "Index built in " << build_time << " seconds" << std::endl;

    // need length of queries to get the right size of the output array
    std::ifstream file(argv[2 + arg_offset]);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file " + std::string(argv[2 + arg_offset]));
    }

    uint32_t num_queries;
    file.read((char*)&num_queries, sizeof(uint32_t));
    file.close();

    uint32_t *out = new uint32_t[num_queries * K];

    index.competition_query(argv[2 + arg_offset], out);

    std::ofstream out_file(argv[3 + arg_offset]);
    if (!out_file.is_open()) {
        throw std::runtime_error("Unable to open file " + std::string(argv[3 + arg_offset]));
    }

    out_file.write((char*)out, num_queries * K * sizeof(uint32_t));
    out_file.close();

    double query_time = t.next_time();

    std::cout << "Build time: " << build_time << " (" << build_time * 100 / (query_time + build_time) << "%)" << std::endl;
    std::cout << "Query time: " << query_time << " (" << query_time * 100 / (query_time + build_time) << "%)" << std::endl;

    std::cout << "Total time: " << t.total_time() << std::endl;

    std::cout << "4M queries would have taken: " << (1600 * query_time + 10 * build_time) / 60 << " minutes" << std::endl;

    if (arg_offset == 1) {
        std::string query_csv_path = "query.csv";
        std::ofstream file(query_csv_path);

        std::ifstream file_exists(query_csv_path);
        if (!file_exists.is_open()) {
            throw std::runtime_error("Unable to open file " + query_csv_path);
        }
        file_exists.seekg(0, file_exists.end);
        size_t file_size = file_exists.tellg();
        file_exists.close();

        if (!file.is_open()) {
            throw std::runtime_error("Unable to open file " + query_csv_path);
        }
        if (file_size == 0) {
            file << "Type 1 QPS, Type 2 QPS, Type 3 QPS, Type 4 QPS, Build Time, Search Time, Timestamp\n";
        }
        std::string csv_row_printing = "";
        for (int i = 0; i < 4; i++) {
            double qps = index.qps_per_case[i];
            file << qps << ",";
            csv_row_printing += std::to_string(qps) + ",";
        }
        file << build_time << "," << query_time << ",";

        std::time_t now = std::time(nullptr);
        char timeStr[20];
        std::strftime(timeStr, sizeof(timeStr), "%Y-%m-%d %H:%M", std::localtime(&now));
        file << timeStr << "\n";
        csv_row_printing += timeStr;

        std::cout << "CSV Row inserted: " << csv_row_printing << std::endl;
        file.close();
    }

    return 0;
}
