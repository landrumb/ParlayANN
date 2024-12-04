/*
  Example usage:
    ./compute_groundtruth -base_path ~/data/sift/sift-1M \
    -query_path ~/data/sift/query-10K -data_type uint8 \
    -dist_func Euclidian -k 100 -gt_path ~/data/sift/GT/sift-1M.gt
*/

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
size_t DATA_BLOCK_SIZE = 100;

struct PriorityQueue {
  size_t k;
  parlay::sequence<pid> pq;

  PriorityQueue(size_t k) : k(k) {
    pq.reserve(k);
    for (size_t i = 0; i < k; i++) {
      pq.push_back(std::make_pair(-1, std::numeric_limits<float>::max()));
    }
  }

  PriorityQueue(parlay::sequence<pid> pq) : k(pq.size()), pq(std::move(pq)) {
    std::sort(pq.begin(), pq.end(), [] (pid a, pid b) {return a.second < b.second;});
  }

  bool insert(pid p) {
    if (p.second > pq[this->k - 1].second) {
      return false;
    }
    pq[this->k - 1] = p;
    std::sort(pq.begin(), pq.end(), [] (pid a, pid b) {return a.second < b.second;});
    return true;
  }

  parlay::sequence<pid> get() {
    return pq;
  }

  void merge(PriorityQueue &other) {
    this->pq.insert(this->pq.end(), other.pq.begin(), other.pq.end());
    std::sort(this->pq.begin(), this->pq.end(), [] (pid a, pid b) {return a.second < b.second;});
    this->pq.resize(this->k);
  }
};

template<typename PointRange>
parlay::sequence<parlay::sequence<pid>> compute_groundtruth_seq(PointRange &B, 
  PointRange &Q, int k){
    unsigned d = B.dimension();
    size_t q = Q.size();
    size_t b = B.size();
    auto answers = parlay::tabulate(q, [&] (size_t i){  
        float topdist = B[0].d_min();   
        int toppos;
        PriorityQueue pq(k);
        for(size_t j = 0; j < b; j++) {
            float dist = Q[i].distance(B[j]);
            pq.insert(std::make_pair((int) j, dist));
        }

        return pq.get();
    });
    std::cout << "Done computing groundtruth" << std::endl;
    return answers;
}

template<typename PointRange>
parlay::sequence<parlay::sequence<pid>> compute_groundtruth_batch(PointRange &B, 
  PointRange &Q, int k, std::pair<int, int> queryRange, std::pair<int, int> dataRange){
    unsigned d = B.dimension();
    size_t q = queryRange.second - queryRange.first;

    // Get the minimum distance possible in this distance metric
    float topdist = B[0].d_min();

    auto answers = parlay::tabulate(q, [&] (size_t i){  
        int toppos;
        PriorityQueue pq(k);
        size_t qIndex = i + queryRange.first;

        for(size_t j=dataRange.first; j<dataRange.second; j++){
            float dist = Q[qIndex].distance(B[j]);
            pq.insert(std::make_pair((int) j, dist));
        }

        return pq.get();
    });
    return answers;
}

template<typename PointRange>
parlay::sequence<parlay::sequence<pid>> compute_groundtruth(PointRange &B, 
  PointRange &Q, int k){
    size_t q = Q.size();
    size_t b = B.size();
    size_t numQueryBlocks = (q + QUERY_BLOCK_SIZE - 1) / QUERY_BLOCK_SIZE;
    size_t numDataBlocks = (b + DATA_BLOCK_SIZE - 1) / DATA_BLOCK_SIZE;
    auto answers = parlay::tabulate(numDataBlocks, [&] (size_t dataBlockIdx){
      size_t start = dataBlockIdx * DATA_BLOCK_SIZE;
      size_t end = std::min((dataBlockIdx + 1) * DATA_BLOCK_SIZE, b);
      auto dataRange = std::make_pair(start, end);
      auto result = parlay::tabulate(numQueryBlocks, [&] (size_t qBlockIdx){
        size_t start = qBlockIdx * QUERY_BLOCK_SIZE;
        size_t end = std::min((qBlockIdx + 1) * QUERY_BLOCK_SIZE, q);
        auto queryRange = std::make_pair(start, end);
        return compute_groundtruth_batch(B, Q, k, queryRange, dataRange);
      }); // result has shape (numQueryBlocks, qBlockSize, k)
      return parlay::flatten(result); // return has shape (q = numQueryBlocks * qBlockSize, k)
    }, 1000000000); // answers has shape (numDataBlocks, q, k)
    
    // auto merged_answers = merge_answers(B, answers, q, k, numDataBlocks);
    // Don't flatten the answers, merge the (q, k) matrices across numDataBlocks
    auto merged_answers = parlay::tabulate(q, [&] (size_t i) {
      PriorityQueue merged_topk(k);
      
      for (size_t dataBlockIdx = 0; dataBlockIdx < numDataBlocks; dataBlockIdx++) {
        PriorityQueue pq(answers[dataBlockIdx][i]);
        merged_topk.merge(pq);
      }
      return merged_topk.get();
    }); // merged_answers has shape (q, k)

    return merged_answers;
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
  DATA_BLOCK_SIZE = P.getOptionIntValue("-data_block_size", 100);

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
