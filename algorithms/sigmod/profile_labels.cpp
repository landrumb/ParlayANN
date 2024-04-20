#include <iostream>
#include <cstdint>
#include <algorithm>
#include <chrono>

#include <string.h>
#include <utility>
#include <unordered_map>

#include "parlay/sequence.h"
#include "parlay/primitives.h"

#include "sigmodindex.h"

int main(int argc, char **argv) {
	if (argc < 2) {
		std::cout << "Usage: " << argv[0] << " [input file] [optional output file]" << std::endl;
		exit(0);
	}

	SigmodIndex<NaiveIndex<float, Euclidian_Point<float>>, NaiveIndex<float, Euclidian_Point<float>>> index;

	auto start_time = std::chrono::high_resolution_clock::now();
	index.load_points(argv[1]);
	auto stop_time = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop_time - start_time);
	std::cout << "Points loaded in " << duration.count() / 1000. << " seconds" << std::endl;

	std::cout << "Num points found: " << index.points.size() << std::endl;

	start_time = std::chrono::high_resolution_clock::now();
	uint32_t min_label = -1, max_label = 0;
	std::unordered_map<uint32_t, uint32_t> label_map;
	for (int i = 0; i < index.points.size(); i++) {
		if (label_map.find(index.labels[i]) == label_map.end()) {
			label_map[index.labels[i]] = 1;
			if (index.labels[i] < min_label) min_label = index.labels[i];
			if (index.labels[i] > max_label) max_label = index.labels[i];
		}
		else {
			label_map[index.labels[i]]++;
		}
	}
	stop_time = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop_time - start_time);
	std::cout << "Labels processed in " << duration.count() / 1000. << " seconds" << std::endl;

	std::cout << "Num unique labels: " << label_map.size() << std::endl;
	std::cout << "Min label: " << min_label << std::endl;
	std::cout << "Max label: " << max_label << std::endl;

	int over100 = 0, sumofover100 = 0;
	auto label_counts = parlay::sequence<std::pair<uint32_t, uint32_t>>::uninitialized(label_map.size());
	int i = 0;
	for (const auto& pair : label_map) {
		label_counts[i].first = pair.first;
		label_counts[i].second = pair.second;
		i++;
		if (pair.second >= 100) {
			over100++;
			sumofover100 += pair.second;
		}
	}

	parlay::integer_sort_inplace(label_counts, [] (std::pair<uint32_t, uint32_t>& p) { return p.second; });
	const int head_size = std::min<int>(10, label_counts.size() / 2);
	std::cout << "Low freq labels:";
	for (int i = 0; i < head_size; i++) {
		std::cout << "\t(" << label_counts[i].first << ", " << label_counts[i].second << ")";
	}
	std::cout << std::endl << "High freq labels:";
	for (int i = 0; i < head_size; i++) {
		std::cout << "\t(" << label_counts[label_counts.size() - i - 1].first << ", " << label_counts[label_counts.size() - i - 1].second << ")";
	}
	std::cout << std::endl;

	if (argc >= 3) {
		std::ofstream writer(argv[2]);
		if (!writer.is_open()) {
			throw std::runtime_error(std::string("Unable to open file ") + argv[2]);
		}
		
		parlay::integer_sort_inplace(label_counts, [] (std::pair<uint32_t, uint32_t>& p) { return p.first; });

		for (auto p : label_counts) {
			writer << p.first << "\t" << p.second << "\n";
		}
		writer.close();
	}
	return 0;
}
