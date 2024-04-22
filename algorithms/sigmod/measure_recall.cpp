#include <iostream>
#include <fstream>
#include <cstdint>
#include <string.h>
#include <cassert>
#include <filesystem>

#define K 100

int main(int argc, char **argv) {
    int arg_offset = 0;

    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <optional CSV flag -r> <output file> <groundtruth file> <optional query file>" << std::endl;
        return 0;
    }

    if (strcmp(argv[1], "-r") == 0) {
        if(argc < 4) {
            std::cout << "Usage: " << argv[0] << " <optional CSV flag -r> <output file> <groundtruth file> <optional query file>" << std::endl;
            return 0;
        } 
        arg_offset = 1; // Skip the csv flag
    }

    std::ifstream out_reader(argv[1 + arg_offset]);
    if (!out_reader.is_open()) {
        throw std::runtime_error("Unable to open file " + std::string(argv[1]));
    }
    std::ifstream gt_reader(argv[2 + arg_offset]);
    if (!gt_reader.is_open()) {
        throw std::runtime_error("Unable to open file " + std::string(argv[2]));
    }

    std::ifstream query_reader;
    if (argc > 3 + arg_offset) {
        query_reader.open(argv[3 + arg_offset]);
        if (!query_reader.is_open()) {
            throw std::runtime_error("Unable to open file " + std::string(argv[3]));
        }
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
    if (argc > 3 + arg_offset) {
        query_reader.ignore(4);
    }

    uint32_t out_buffer[K];
    uint32_t gt_buffer[K];
    uint64_t total_correct = 0;
    for (int i = 0; i < num_queries; i++) {
        uint32_t problem_type;
        if (argc > 3 + arg_offset) {
            float problem_type_buffer;
            query_reader.read((char*)&problem_type_buffer, sizeof(uint32_t));
            query_reader.ignore(103 * sizeof(uint32_t));
            problem_type = (uint32_t)problem_type_buffer;
            assert(problem_type < 4);
            problem_type_count[problem_type]++;
        }

        out_reader.read((char*)&out_buffer[0], K * sizeof(uint32_t));
        gt_reader.read((char*)&gt_buffer[0], K * sizeof(uint32_t));
        for (int j = 0; j < K; j++) {
            for (int k = 0; k < K; k++) {
                if (out_buffer[j] == gt_buffer[k]) {
                    total_correct++;
                    if (argc > 3) {
                        correct_count[problem_type]++;
                    }
                    break;
                }
            }
        }
    }

    if (argc > 3 + arg_offset) {
        for (int i = 0; i < 4; i++) {
            std::cout << "Recall for problems of type " << i << ": " << (double)correct_count[i] / (problem_type_count[i] * K) << std::endl;
        }
    }
    
    // Output to CSV if -r was included (right now there is only the -r flag,
    // so this conditional is a bit of a hack)
    if (arg_offset == 1) {
        std::string recall_csv_path = "recall.csv";
        std::ofstream file(recall_csv_path);
        std::ifstream file_exists(recall_csv_path);
        if (!file_exists.is_open()) {
            throw std::runtime_error("Unable to open file " + recall_csv_path);
        }
        file_exists.seekg(0, file_exists.end);
        size_t file_size = file_exists.tellg();
        file_exists.close();

        if (!file.is_open()) {
            throw std::runtime_error("Unable to open file " + recall_csv_path);
        }
        if (file_size == 0) {
            file << "Type 1 Recall, Type 2 Recall, Type 3 Recall, Type 4 Recall, Timestamp\n";
        }
        std::string csv_row_printing = "";
        for (int i = 0; i < 4; i++) {
            double recall = (double)correct_count[i] / (problem_type_count[i] * K);
            file << recall << ",";
            csv_row_printing += std::to_string(recall) + ",";
        }

        std::time_t now = std::time(nullptr);
        char timeStr[20];
        std::strftime(timeStr, sizeof(timeStr), "%Y-%m-%d %H:%M", std::localtime(&now));
        file << timeStr << "\n";
        csv_row_printing += timeStr;

        std::cout << "CSV Row inserted: " << csv_row_printing << std::endl;
    }

    std::cout << "Total recall: " << (double)total_correct / (num_queries * K) << std::endl;
    return 0;
}
