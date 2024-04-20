ifeq (, $(shell which jemalloc-config))
JEMALLOC =
else
JEMALLOCLD = $(shell jemalloc-config --libdir)
JEMALLOC = -L$(JEMALLOCLD) -ljemalloc 
endif

CXXFLAGS = -mcx16 -O3 -std=c++17 -march=native -DNDEBUG -I .
LDFLAGS = -ldl $(JEMALLOC)

OMPFLAGS = -DPARLAY_OPENMP -fopenmp
CILKFLAGS = -DPARLAY_CILK -fcilkplus
PBBFLAGS = -DHOMEGROWN -pthread

ifdef OPENMP
CXX = g++
CFLAGS = $(OMPFLAGS) $(CXXFLAGS)
LFLAGS = $(OMPFLAGS) $(LDFLAGS)

else ifdef CILK
CXX = g++
CFLAGS = $(CILKFLAGS) $(CXXFLAGS)
LFLAGS = $(CILKFLAGS) $(LDFLAGS)

else
CXX = g++
CFLAGS = $(PBBFLAGS) $(CXXFLAGS)
LFLAGS = $(PBBFLAGS) $(LDFLAGS)
endif


TARGETS = profile_dims
INCLUDE = ../../parlaylib/include
REQUIRE = sigmodindex.h ../utils/beamSearch.h ../utils/check_nn_recall.h ../utils/NSGDist.h ../utils/parse_results.h ../utils/graph.h ../utils/point_range.h

all: $(TARGETS)

%: %.cpp $(REQUIRE)
	$(CXX) -DSTATS $(CFLAGS) -I $(INCLUDE) -o $@ $< $(LFLAGS)

clean:
	rm -f $(TARGETS)

