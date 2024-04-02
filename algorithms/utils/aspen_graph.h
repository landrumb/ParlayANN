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

#pragma once

#include <algorithm>
#include <iostream>

#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/internal/file_map.h"
#include "NSGDist.h"
#include "types.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>


#include <cpam/cpam.h>
#include <pam/pam.h>

#include <mutex>

#include "aspen/aspen.h"



template<typename indexType>
struct Aspen_Graph{
    struct empty_weight {};
    using GraphT = aspen::symmetric_graph<empty_weight, indexType>;
    using vertex = typename GraphT::vertex;
    using edge_tree = typename GraphT::edge_tree;
    using vertex_tree = typename GraphT::vertex_tree;
    using vertex_node = typename GraphT::vertex_node;
    using version = typename aspen::versioned_graph<GraphT>::version;

    using iT = indexType;

    struct Aspen_Vertex{

        size_t size(){return v.out_degree();}
        indexType id(){return v.id;}

        Aspen_Vertex(){}
        Aspen_Vertex(vertex v, long maxDeg, GraphT &G) : v(v), maxDeg(maxDeg), G(G) {}

        template<typename rangeType>
        void append_neighbors(const rangeType &r){
            parlay::sequence<indexType> neighbors;
            auto f = [&](indexType u, indexType neighbor_id, empty_weight weight) {
                neighbors.push_back(neighbor_id);
                return true;
            };
            v.out_neighbors().foreach_cond(f);
            for(indexType i : r) neighbors.push_back(i);
            return update_neighbors(neighbors);
        }
        
        template<typename rangeType>
        void update_neighbors(const rangeType &r){
            if(r.size() > maxDeg){
                std::cout << "Error in update: tried to exceed max degree" << std::endl;
                abort();
            }
            auto begin = (std::tuple<indexType, empty_weight>*)(r.begin());
            auto tree = edge_tree(begin, begin + r.size());
            auto seq = {std::make_tuple(v.id, std::move(tree))};
            G.insert_vertices_batch(seq.size(), seq.begin());
        }

        parlay::sequence<indexType> neighbors(){
            parlay::sequence<indexType> neighbors;
            auto f = [&](indexType u, indexType neighbor_id, empty_weight weight) {
                neighbors.push_back(neighbor_id);
                return true;
            };
            v.out_neighbors().foreach_cond(f);
            return neighbors;
        }

        //prefetch and reorder not supported for this graph type
        template<typename F>
        void reorder(F&& f){}
        void prefetch(){}

        private:
            vertex v;
            long maxDeg;
            GraphT &G;

    };

    struct Graph{

        long max_degree() const {return maxDeg;}
        size_t size() const {return V.graph.num_vertices();}

        Graph(){}

        Graph(long maxDeg) : maxDeg(maxDeg) {}

        Graph(version V, long maxDeg, bool read_only = true) : V(V), maxDeg(maxDeg) {
            if(read_only) G = V.graph;
            else G = (V.graph).functional_copy();
        }

        void batch_update(parlay::sequence<std::pair<indexType, parlay::sequence<indexType>>> &edges){
            // std::cout << "processing updates to " << edges.size() << " vertices" << std::endl;
            auto vals = parlay::tabulate(edges.size(), [&] (size_t i){
                indexType index = edges[i].first;
                size_t ngh_size = edges[i].second.size();
                if(ngh_size > maxDeg){
                    std::cout << "ERROR in batch_update: ngh too large" << std::endl;
                    abort();
                }
                auto begin = (std::tuple<indexType, empty_weight>*)(edges[i].second.begin());
                auto tree = edge_tree(begin, begin + ngh_size);
                return std::make_tuple(index, std::move(tree));
            });
            G.insert_vertices_batch(vals.size(), vals.begin());
        }

        void batch_delete(parlay::sequence<indexType> &deletes){
            G.delete_vertices_batch(deletes.size(), deletes.begin());
        }

        GraphT move_graph(){return std::move(G);}

        version move_version(){return std::move(V);}

        Aspen_Vertex operator [] (indexType i) {return Aspen_Vertex(G.get_vertex(i), maxDeg, G);}

        void save(char* oFile){
            size_t n = V.graph.num_vertices();
            std::cout << "Writing graph with " << n << " points and max degree " << maxDeg
                        << std::endl;
            parlay::sequence<indexType> preamble = {static_cast<indexType>(n), static_cast<indexType>(maxDeg)};
            parlay::sequence<indexType> sizes = parlay::tabulate(n, [&] (size_t i){return static_cast<indexType>((*this)[i].size());});
            std::ofstream writer;
            writer.open(oFile, std::ios::binary | std::ios::out);
            writer.write((char*)preamble.begin(), 2 * sizeof(indexType));
            writer.write((char*)sizes.begin(), sizes.size() * sizeof(indexType));
            size_t BLOCK_SIZE = 1000000;
            size_t index = 0;
            while(index < n){
                size_t floor = index;
                size_t ceiling = index+BLOCK_SIZE <= n ? index+BLOCK_SIZE : n;
                parlay::sequence<parlay::sequence<indexType>> edge_data = parlay::tabulate(ceiling-floor, [&] (size_t i){
                    return parlay::tabulate(sizes[i+floor], [&] (size_t j){return (*this)[i+floor].neighbors()[j];});
                });
                parlay::sequence<indexType> data = parlay::flatten(edge_data);
                writer.write((char*)data.begin(), data.size() * sizeof(indexType));
                index = ceiling;
            }
            writer.close();
        }

        private:
            long maxDeg;
            version V;
            GraphT G;
    };

    Aspen_Graph(){}

    Aspen_Graph(long md, size_t n) : maxDeg(md){
        GraphT GG;
        VG = aspen::versioned_graph<GraphT>(std::move(GG));
    }

    Aspen_Graph(char* gFile){
        GraphT GG;
        std::ifstream reader(gFile);
        assert(reader.is_open());

        //read num points and max degree
        indexType num_points;
        indexType max_deg;
        reader.read((char*)(&num_points), sizeof(indexType));
        size_t n = num_points;
        reader.read((char*)(&max_deg), sizeof(indexType));
        maxDeg = max_deg;
        std::cout << "Detected " << num_points << " points with max degree " << max_deg << std::endl;

        //read degrees and perform scan to find offsets
        indexType* degrees_start = new indexType[n];
        reader.read((char*)(degrees_start), sizeof(indexType)*n);
        indexType* degrees_end = degrees_start + n;
        parlay::slice<indexType*, indexType*> degrees0 = parlay::make_slice(degrees_start, degrees_end);
        auto degrees = parlay::tabulate(degrees0.size(), [&] (size_t i){return static_cast<size_t>(degrees0[i]);});
        auto [offsets, total] = parlay::scan(degrees);
        std::cout << "Total: " << total << std::endl;
        offsets.push_back(total);


        //write 1000000 vertices at a time
        size_t BLOCK_SIZE=1000000;
        size_t index = 0;
        size_t total_size_read = 0;
        while(index < n){
            size_t g_floor = index;
            size_t g_ceiling = g_floor + BLOCK_SIZE <= n ? g_floor + BLOCK_SIZE : n;
            size_t total_size_to_read = offsets[g_ceiling]-offsets[g_floor];
            indexType* edges_start = new indexType[total_size_to_read];
            reader.read((char*)(edges_start), sizeof(indexType)*total_size_to_read);
            indexType* edges_end = edges_start + total_size_to_read;
            parlay::slice<indexType*, indexType*> edges = parlay::make_slice(edges_start, edges_end);
            auto updates = parlay::tabulate(g_ceiling-g_floor, [&] (size_t i){
                indexType index = g_floor+i;
                parlay::sequence<indexType> nbh = parlay::tabulate(degrees[index], [&] (size_t j){
                    return edges[offsets[index] - total_size_read + j];
                });
                return std::make_pair(index, nbh);
            });
            auto vals = parlay::tabulate(updates.size(), [&](size_t i) {
                indexType index = updates[i].first;
                size_t ngh_size = updates[i].second.size();
                auto begin = (std::tuple<indexType, empty_weight>*)(updates[i].second.begin());
                auto tree = edge_tree(begin, begin + ngh_size);
                return std::make_tuple(index, std::move(tree));
            });
            GG.insert_vertices_batch(vals.size(), vals.begin());
            total_size_read += total_size_to_read;
            index = g_ceiling; 
            delete[] edges_start;
        }
        delete[] degrees_start;
        VG = aspen::versioned_graph<GraphT>(std::move(GG));
    }

    Graph Get_Graph(){
        auto S = VG.acquire_version();
        std::cout << "Acquired writable version with timestamp " << S.timestamp << std::endl;
        return Graph(std::move(S), maxDeg, false);
    }

    //TODO add safeguard to avoid updating graph in read-only mode
    Graph Get_Graph_Read_Only(){
        auto S = VG.acquire_version();
        std::cout << "Acquired read-only version with timestamp " << S.timestamp << std::endl;
        return Graph(std::move(S), maxDeg, true);
    }

    //TODO do we need to do anything with the copy of graph that's stored in the graph wrapper?
    void Release_Graph(Graph G){
        auto S = G.move_version();
        VG.release_version(std::move(S));
    }

    void Update_Graph(Graph G){
        auto S = G.move_version();
        GraphT new_G = G.move_graph();
        VG.add_version_from_graph(std::move(new_G));
        VG.release_version(std::move(S));
    }

    void save(char* oFile) {
        auto S = VG.acquire_version();
        std::cout << "Acquired read-only version with timestamp " << S.timestamp
                << std::endl;
        Graph GG = Graph(std::move(S), maxDeg, true);
        GG.save(oFile);
    }


    private:
        aspen::versioned_graph<GraphT> VG;
        size_t maxDeg;

};