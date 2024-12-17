#include <iostream>
#include <algorithm>
#include <cstdint>
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/io.h"
#include "utils/euclidian_point.h"
#include "utils/mips_point.h"
#include "utils/point_range.h"
#include "../algorithms/bench/parse_command_line.h"

using pid = std::pair<int, float>;
using namespace parlayANN;

size_t QUERY_BLOCK_SIZE = 100;
size_t DATA_BLOCK_SIZE = 5000;

struct PriorityQueue {
  using compare_desc = std::function<bool(const pid&, const pid&)>;

  size_t k;
  // store elements as a "max-heap" so the largest distance is at the front
  // this way insertion is O(log k) instead of a full O(k log k) sort.
  // at the end, we can sort ascending if needed.
  parlay::sequence<pid> pq;

  struct max_cmp {
    bool operator()(const pid &a, const pid &b) const {
      // bigger distance => "less" => top of the heap
      return a.second < b.second;
    }
  };

  PriorityQueue(size_t k) : k(k) {
    pq.resize(k, std::make_pair(-1, std::numeric_limits<float>::max()));
    // build a max-heap of size k, all distances initialized to infinity
    std::make_heap(pq.begin(), pq.end(), max_cmp());
  }

  PriorityQueue(parlay::sequence<pid> arr) : k(arr.size()), pq(std::move(arr)) {
    // build max-heap in-place
    std::make_heap(pq.begin(), pq.end(), max_cmp());
  }

  bool insert(pid p) {
    // if p is worse (i.e. distance >= largest distance in the heap) skip
    if (p.second >= pq.front().second) return false;
    // otherwise pop the worst element, add p, push back up
    std::pop_heap(pq.begin(), pq.end(), max_cmp());
    pq.back() = p;
    std::push_heap(pq.begin(), pq.end(), max_cmp());
    return true;
  }

  // returning a sorted ascending sequence
  parlay::sequence<pid> get() const {
    parlay::sequence<pid> out = pq; 
    std::sort(out.begin(), out.end(), [] (pid a, pid b) {return a.second < b.second;});
    return out;
  }

  void merge(PriorityQueue &other) {
    // merge by inserting each element from 'other' 
    for (auto &x : other.pq) {
      insert(x);
    }
  }
};


template<typename PointRange>
parlay::sequence<parlay::sequence<pid>> compute_groundtruth(PointRange &B, PointRange &Q, int k) {
  size_t q = Q.size();
  size_t b = B.size();
  size_t numDataBlocks = (b + DATA_BLOCK_SIZE - 1) / DATA_BLOCK_SIZE;

  // partial results for each query
  parlay::sequence<PriorityQueue> partialResults(q, PriorityQueue(k));

  // serial loop over data blocks
  for (size_t dataBlockIdx = 0; dataBlockIdx < numDataBlocks; dataBlockIdx++) {
    size_t start = dataBlockIdx * DATA_BLOCK_SIZE;
    size_t end = std::min((dataBlockIdx + 1) * DATA_BLOCK_SIZE, b);

    // parallel for loop over all queries
    parlay::parallel_for(0, q, [&](size_t i) {
      for (size_t j = start; j < end; j++) {
        float dist = Q[i].distance(B[j]);
        partialResults[i].insert(std::make_pair((int) j, dist));
      }
    });
  }

  // finalize and gather results
  parlay::sequence<parlay::sequence<pid>> answers(q);
  parlay::parallel_for(0, q, [&](size_t i){
    answers[i] = partialResults[i].get();
  });

  std::cout << "done computing groundtruth" << std::endl;
  return answers;
}


// ibin is the same as the binary groundtruth format used in the
// big-ann-benchmarks (see: https://big-ann-benchmarks.com/neurips21.html)
void write_ibin(parlay::sequence<parlay::sequence<pid>> &result, const std::string outFile, int k){
    std::cout << "Writing file with dimension " << result[0].size() << std::endl;
    std::cout << "File contains groundtruth for " << result.size() << " query points" << std::endl;

    auto less = [&] (pid a, pid b) {return a.second < b.second;};
    parlay::sequence<int> preamble = {static_cast<int>(result.size()), static_cast<int>(result[0].size())};
    size_t n = result.size();
    parlay::parallel_for(0, result.size(), [&] (size_t i){
      parlay::sort_inplace(result[i], less);
    });
    auto ids = parlay::tabulate(result.size(), [&] (size_t i){
        parlay::sequence<int> data;
        for(int j=0; j<k; j++){
          data.push_back(static_cast<int>(result[i][j].first));
        }
        return data;
    });
    auto distances = parlay::tabulate(result.size(), [&] (size_t i){
        parlay::sequence<float> data;
        for(int j=0; j<k; j++){
          data.push_back(static_cast<float>(result[i][j].second));
        }
        return data;
    });
    parlay::sequence<int> flat_ids = parlay::flatten(ids);
    parlay::sequence<float> flat_dists = parlay::flatten(distances);

    auto pr = preamble.begin();
    auto id_data = flat_ids.begin();
    auto dist_data = flat_dists.begin();
    std::ofstream writer;
    writer.open(outFile, std::ios::binary | std::ios::out);
    writer.write((char *) pr, 2*sizeof(int));
    writer.write((char *) id_data, n * k * sizeof(int));
    writer.write((char *) dist_data, n * k * sizeof(float));
    writer.close();
}

int main(int argc, char* argv[]) {
  commandLine P(argc,argv,
    "[-base_path <b>] [-query_path <q>] "
    "[-data_type <d>] [-k <k> ] [-dist_func <d>] [-gt_path <outfile>] [-query_block_size <qbs>] [-data_block_size <dbs>]");

  char* gFile = P.getOptionValue("-gt_path");
  char* qFile = P.getOptionValue("-query_path");
  char* bFile = P.getOptionValue("-base_path");
  char* vectype = P.getOptionValue("-data_type");
  char* dfc = P.getOptionValue("-dist_func");
  int k = P.getOptionIntValue("-k", 100);
  QUERY_BLOCK_SIZE = P.getOptionIntValue("-query_block_size", 100);
  DATA_BLOCK_SIZE = P.getOptionIntValue("-data_block_size", 5000);

  std::string df = std::string(dfc);
  if(df != "Euclidian" && df != "mips"){
    std::cout << "Error: invalid distance type: specify Euclidian or mips" << std::endl;
    abort();
  }

  std::string tp = std::string(vectype);
  if((tp != "uint8") && (tp != "int8") && (tp != "float")){
    std::cout << "Error: data type not specified correctly, specify int8, uint8, or float" << std::endl;
    abort();
  }

  std::cout << "Computing the " << k << " nearest neighbors" << std::endl;

  int maxDeg = 0;

  parlay::sequence<parlay::sequence<pid>> answers;
  std::string base = std::string(bFile);
  std::string query = std::string(qFile);

  if(tp == "float"){
    std::cout << "Detected float coordinates" << std::endl;
    if(df == "Euclidian"){
      auto B = PointRange<Euclidian_Point<float>>(bFile);
      auto Q = PointRange<Euclidian_Point<float>>(qFile);
      answers = compute_groundtruth<PointRange<Euclidian_Point<float>>>(B, Q, k);
    } else if(df == "mips"){
      auto B = PointRange<Mips_Point<float>>(bFile);
      auto Q = PointRange<Mips_Point<float>>(qFile);
      answers = compute_groundtruth<PointRange<Mips_Point<float>>>(B, Q, k);
    }
  }else if(tp == "uint8"){
    std::cout << "Detected uint8 coordinates" << std::endl;
    if(df == "Euclidian"){
      auto B = PointRange<Euclidian_Point<uint8_t>>(bFile);
      auto Q = PointRange<Euclidian_Point<uint8_t>>(qFile);
      answers = compute_groundtruth<PointRange<Euclidian_Point<uint8_t>>>(B, Q, k);
    } else if(df == "mips"){
      auto B = PointRange<Mips_Point<uint8_t>>(bFile);
      auto Q = PointRange<Mips_Point<uint8_t>>(qFile);
      answers = compute_groundtruth<PointRange<Mips_Point<uint8_t>>>(B, Q, k);
    }
  } else if(tp == "int8"){
    std::cout << "Detected int8 coordinates" << std::endl;
    if(df == "Euclidian"){
      auto B = PointRange<Euclidian_Point<int8_t>>(bFile);
      auto Q = PointRange<Euclidian_Point<int8_t>>(qFile);
      answers = compute_groundtruth<PointRange<Euclidian_Point<int8_t>>>(B, Q, k);
    } else if(df == "mips"){
      auto B = PointRange<Mips_Point<int8_t>>(bFile);
      auto Q = PointRange<Mips_Point<int8_t>>(qFile);
      answers = compute_groundtruth<PointRange<Mips_Point<int8_t>>>(B, Q, k);
    }
  }
  write_ibin(answers, std::string(gFile), k);

  return 0;
}
