#include <iostream>
#include <fstream>
#include <cstdint>
#include <string.h>
#include <cassert>

#define K 100

int main(int argc, char **argv) {
    if (argc < 4) {
        std::cout << "Usage: " << argv[0] << " <output file> <groundtruth file> <query file>" << std::endl;
        return 0;
    }

    std::ifstream out_reader(argv[1]);
    if (!out_reader.is_open()) {
        throw std::runtime_error("Unable to open file " + std::string(argv[1]));
    }
    std::ifstream gt_reader(argv[2]);
    if (!gt_reader.is_open()) {
        throw std::runtime_error("Unable to open file " + std::string(argv[2]));
    }
    std::ifstream query_reader(argv[3]);
    if (!query_reader.is_open()) {
        throw std::runtime_error("Unable to open file " + std::string(argv[3]));
    }

    out_reader.seekg(0, out_reader.end);
    gt_reader.seekg(0, gt_reader.end);
    if (out_reader.tellg() != gt_reader.tellg()) {
        throw std::runtime_error("Output and groundtruth files are different sizes");
    }

    uint64_t num_bytes = out_reader.tellg();
    if (num_bytes % (K * sizeof(uint32_t)) != 0) {
        throw std::runtime_error("Number of bytes is not divisible by 4 * " + std::to_string(K));
    }
    uint32_t num_queries = num_bytes / (K * sizeof(uint32_t));

    out_reader.seekg(0, out_reader.beg);
    gt_reader.seekg(0, gt_reader.beg);

    uint32_t problem_type_count[4] = {0, 0, 0, 0};
    uint32_t correct_count[4] = {0, 0, 0, 0};
    query_reader.ignore(4);

    uint32_t out_buffer[K];
    uint32_t gt_buffer[K];
    uint64_t total_correct = 0;
    for (int i = 0; i < num_queries; i++) {
        uint32_t problem_type;
        float problem_type_buffer;
        query_reader.read((char*)&problem_type_buffer, sizeof(uint32_t));
        query_reader.ignore(103 * sizeof(uint32_t));
        problem_type = (uint32_t)problem_type_buffer;
        assert(problem_type < 4);
        problem_type_count[problem_type]++;
        out_reader.read((char*)&out_buffer[0], K * sizeof(uint32_t));
        gt_reader.read((char*)&gt_buffer[0], K * sizeof(uint32_t));
        for (int j = 0; j < K; j++) {
            for (int k = 0; k < K; k++) {
                if (out_buffer[j] == gt_buffer[k]) {
                    total_correct++;
                    correct_count[problem_type]++;
                    break;
                }
            }
        }
    }

    for (int i = 0; i < 4; i++) {
        std::cout << "Recall for problems of type " << i << ": " << (double)correct_count[i] / (problem_type_count[i] * K) << std::endl;
    }

    std::cout << "Total recall: " << (double)total_correct / (num_queries * K) << std::endl;
    return 0;
}
