#include <iostream>
#include <cstdint>
#include <algorithm>
#include <chrono>

#include <cstring>
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

	SigmodIndex<int, int> index;

	auto start_time = std::chrono::high_resolution_clock::now();
	index.load_points(argv[1]);
	auto stop_time = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop_time - start_time);
	std::cout << "Points loaded in " << duration.count() / 1000. << " seconds" << std::endl;

	std::cout << "Num points found: " << index.points.size() << std::endl;

	const int displayed_buckets = 100, written_buckets = 1000, displayed_height = 20;
	uint32_t displayed_hist[displayed_buckets];
	uint32_t written_hist[written_buckets];
	float min_timestamp = 1, max_timestamp = 0;
	int max_freq = 0;



	start_time = std::chrono::high_resolution_clock::now();
	std::memset(&displayed_hist[0], 0, displayed_buckets * sizeof(uint32_t));
	std::memset(&written_hist[0], 0, written_buckets * sizeof(uint32_t));
	for (int i = 0; i < index.points.size(); i++) {
		int j = (int)(index.timestamps[i] * displayed_buckets);
		displayed_hist[j]++;
		if (displayed_hist[j] > displayed_hist[max_freq]) max_freq = j;
		written_hist[(int)(index.timestamps[i] * written_buckets)]++;
		if (index.timestamps[i] < min_timestamp) min_timestamp = index.timestamps[i];
		if (index.timestamps[i] > max_timestamp) max_timestamp = index.timestamps[i];
	}
	stop_time = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop_time - start_time);
	std::cout << "Timestamps processed in " << duration.count() / 1000. << " seconds" << std::endl;

	std::cout << "Min timestamp: " << min_timestamp << std::endl;
	std::cout << "Max timestamp: " << max_timestamp << std::endl;
	std::cout << "Most frequent time range is " << ((float)max_freq / displayed_buckets) << "-" << ((float)(max_freq + 1) / displayed_buckets) << " with " << displayed_hist[max_freq] << " timestamps" << std::endl;
	
	for (int i = displayed_height - 1; i >= 0; i--) {
		for (int j = 0; j < displayed_buckets; j++) {
			if (displayed_hist[j] >= displayed_hist[max_freq] * i / 20) std::cout << "-";
			else std::cout << " ";
		}
		std::cout << std::endl;
	}

	if (argc >= 3) {
		std::ofstream writer(argv[2]);
		if (!writer.is_open()) {
			throw std::runtime_error(std::string("Unable to open file ") + argv[2]);
		}

		for (int i = 0; i < written_buckets; i++) {
			writer << i << "\t" << written_hist[i] << "\n";
		}
		writer.close();
	}
	return 0;
}
