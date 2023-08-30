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
#include <set>

#include "beamSearch.h"
#include "csvfile.h"
#include "parse_results.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "types.h"

template<typename T, template<typename C> class Point, template<typename E, template<typename D> class P> class PointRange>
nn_result checkRecall(
        Graph<unsigned int> &G,
        PointRange<T, Point> &Base_Points,
        PointRange<T, Point> &Query_Points,
        groundTruth<int> GT,
        int k, int beamQ, float cut, bool random,
        int limit, int start_point, int r) {
  if (GT.size() > 0 && r > GT.dimension()) {
    std::cout << r << "@" << r << " too large for ground truth data of size "
              << GT.dimension() << std::endl;
    abort();
  }
  // std::cout << "here" << std::endl;

  parlay::sequence<parlay::sequence<int>> all_ngh;

  parlay::internal::timer t;
  float query_time;
  if(random){
    all_ngh = beamSearchRandom(Query_Points, G, Base_Points, beamQ, k, cut, limit);
    t.next_time();
    all_ngh = beamSearchRandom(Query_Points, G, Base_Points, beamQ, k, cut, limit);
    query_time = t.next_time();
  }else{
    all_ngh = searchAll(Query_Points, G, Base_Points, beamQ, k, start_point, cut, limit);
    t.next_time();
    all_ngh = searchAll(Query_Points, G, Base_Points, beamQ, k, start_point, cut, limit);
    query_time = t.next_time();
  }

  float recall = 0.0;
  //TODO deprecate this after further testing
  bool dists_present = true;
  // bool dists_present = false;
  if (GT.size() > 0 && !dists_present) {
    size_t n = Query_Points.size();
    int numCorrect = 0;
    for (int i = 0; i < n; i++) {
      std::set<int> reported_nbhs;
      for (int l = 0; l < r; l++) reported_nbhs.insert((all_ngh[i])[l]);
      for (int l = 0; l < r; l++) {
        if (reported_nbhs.find((GT.coordinates(i,l))) !=
            reported_nbhs.end()) {
          numCorrect += 1;
        }
      }
    }
    recall = static_cast<float>(numCorrect) / static_cast<float>(r * n);
  } else if (GT.size() > 0 && dists_present) {
    size_t n = Query_Points.size();
    
    int numCorrect = 0;
    for (int i = 0; i < n; i++) {
      // std::cout << (GT.size() == Query_Points.size()) << std::endl;
      // std::cout << i << std::endl;
      parlay::sequence<int> results_with_ties;
      for (int l = 0; l < r; l++)
        results_with_ties.push_back(GT.coordinates(i,l));
      // std::cout << "here1" << std::endl;
      float last_dist = GT.distances(i, r-1);
      for (int l = r; l < GT.dimension(); l++) {
        if (GT.distances(i,l) == last_dist) {
          results_with_ties.push_back(GT.coordinates(i,l));
        }
      }
      // std::cout << "here2" << std::endl;
      std::set<int> reported_nbhs;
      for (int l = 0; l < r; l++) reported_nbhs.insert((all_ngh[i])[l]);
      for (int l = 0; l < results_with_ties.size(); l++) {
        if (reported_nbhs.find(results_with_ties[l]) != reported_nbhs.end()) {
          numCorrect += 1;
        }
      }
    }
    recall = static_cast<float>(numCorrect) / static_cast<float>(r * n);
  }
  float QPS = Query_Points.size() / query_time;
  //TODO add back stats
  // auto stats = query_stats(q);
  parlay::sequence<size_t> stats = {0,0,0,0};
  nn_result N(recall, stats, QPS, k, beamQ, cut, Query_Points.size(), limit, r);
  return N;
}

void write_to_csv(std::string csv_filename, parlay::sequence<float> buckets,
                  parlay::sequence<nn_result> results, Graph_ G) {
  csvfile csv(csv_filename);
  csv << "GRAPH"
      << "Parameters"
      << "Size"
      << "Build time"
      << "Avg degree"
      << "Max degree" << endrow;
  csv << G.name << G.params << G.size << G.time << G.avg_deg << G.max_deg
      << endrow;
  csv << endrow;
  csv << "Num queries"
      << "Target recall"
      << "Actual recall"
      << "QPS"
      << "Average Cmps"
      << "Tail Cmps"
      << "Average Visited"
      << "Tail Visited"
      << "k"
      << "Q"
      << "cut" << endrow;
  for (int i = 0; i < results.size(); i++) {
    nn_result N = results[i];
    csv << N.num_queries << buckets[i] << N.recall << N.QPS << N.avg_cmps
        << N.tail_cmps << N.avg_visited << N.tail_visited << N.k << N.beamQ
        << N.cut << endrow;
  }
  csv << endrow;
  csv << endrow;
}

parlay::sequence<int> calculate_limits(size_t avg_visited) {
  parlay::sequence<int> L(9);
  for (float i = 1; i < 10; i++) {
    L[i - 1] = (int)(i * ((float)avg_visited) * .1);
  }
  auto limits = parlay::remove_duplicates(L);
  return limits;
}

template<typename T, template<typename C> class Point, template<typename E, template<typename D> class P> class PointRange>
void search_and_parse(Graph_ G_, Graph<unsigned int> &G, PointRange<T, Point> &Base_Points,
   PointRange<T, Point> &Query_Points, 
  groundTruth<int> GT, char* res_file, 
  bool random=true, uint start_point=0){

  parlay::sequence<nn_result> results;
  std::vector<int> beams;
  std::vector<int> allk;
  std::vector<int> allr;
  std::vector<float> cuts;
  if (G.size() <= 200000) {
    beams = {15, 20, 30, 50, 75, 100};
    allk = {10, 15, 20, 50};
    allr = {10, 20};
    cuts = {1.1, 1.15, 1.2, 1.25};
  } else {
    beams = {15, 20, 30, 50, 75, 100, 150, 200, 250, 375, 500, 1000};
    allk = {10, 15, 20, 30, 50, 100};
    allr = {10};  // {10, 20, 100};
    cuts = {1.15, 1.25, 1.35};
  }

    for (int r : allr) {
      results.clear();
      for (float cut : cuts)
	for (float Q : beams)
	  if (Q > r)
	    results.push_back(checkRecall(G, Base_Points, Query_Points, GT, r, Q, cut, random, -1, start_point, r));

      // check "limited accuracy"
      // TODO re-instate this after adding stats counters
      parlay::sequence<int> limits = calculate_limits(results[0].avg_visited);
      for(int l : limits)
	results.push_back(checkRecall(G, Base_Points, Query_Points, GT, r, r+5, 1.15, random, l, start_point, r));

      // check "best accuracy"
      if(G.size() <= 200000) results.push_back(checkRecall(G, Base_Points, Query_Points, GT, r, 500, 10.0, random, -1, start_point, r));
      else results.push_back(checkRecall(G, Base_Points, Query_Points, GT, 100, 1000, 10.0, random, -1, start_point, r));

    parlay::sequence<float> buckets = {
        .1, .15, .2,  .25, .3,  .35,  .4,   .45,   .5,   .55,
        .6, .65, .7,  .73, .75, .77,  .8,   .83,   .85,  .87,
        .9, .93, .95, .97, .99, .995, .999, .9995, .9999};
    auto [res, ret_buckets] = parse_result(results, buckets);
    std::cout << std::endl;
    if (res_file != NULL)
      write_to_csv(std::string(res_file), ret_buckets, res, G_);
  }
}
