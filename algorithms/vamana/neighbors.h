// This code is part of the Problem Based Benchmark Suite (PBBS)
// Copyright (c) 2011 Guy Blelloch and the PBBS team
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights (to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <algorithm>

#include "../utils/NSGDist.h"
#include "../utils/beamSearch.h"
#include "../utils/check_nn_recall.h"
#include "../utils/indexTools.h"
#include "../utils/parse_results.h"
#include "../utils/stats.h"
#include "../utils/types.h"
#include "index.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/random.h"

// void ANN(Graph G, Data D, BuildParams B, Query_Data Q, groundTruth GT, char* res_file)




template <typename T>
void ANN(parlay::sequence<Tvec_point<T> *> &v, int k, int maxDeg, int beamSize,
         int beamSizeQ, double alpha, double dummy,
         parlay::sequence<Tvec_point<T> *> &q,
         parlay::sequence<ivec_point> &groundTruth, char *res_file,
         bool graph_built, Distance *D, data_store<T> &Data) {
  parlay::internal::timer t("ANN");
  unsigned d = (v[0]->coordinates).size();
  using findex = knn_index<T>;
  findex I(maxDeg, beamSize);
  double idx_time;
  if(graph_built){
    idx_time = 0;
  } else{
    parlay::sequence<int> inserts = parlay::tabulate(v.size(), [&] (size_t i){
					    return static_cast<int>(i);});
    I.build_index(v, std::move(inserts), Data);
    idx_time = t.next_time();
  }

  int medoid = I.get_medoid();
  std::string name = "Vamana";
  std::string params =
      "R = " + std::to_string(maxDeg) + ", L = " + std::to_string(beamSize);
  auto [avg_deg, max_deg] = graph_stats(v);
  auto vv = visited_stats(v);
  std::cout << "Average visited: " << vv[0] << ", Tail visited: " << vv[1]
            << std::endl;
  Graph G(name, params, v.size(), avg_deg, max_deg, idx_time);
  G.print();
  if(q.size() != 0) search_and_parse(G, v, Data, q, groundTruth, res_file, false, medoid);
}



