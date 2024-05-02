TARGETS = query measure_recall
INCLUDE = ../../parlaylib/include
REQUIRE = sigmodindex.h ../utils/beamSearch.h ../utils/check_nn_recall.h ../utils/NSGDist.h ../utils/parse_results.h ../utils/graph.h ../utils/point_range.h small_index.h big_index.h filter_ivf_index.h 


#ifeq (, $(shell which jemalloc-config))
#JEMALLOC =
#else
#JEMALLOCLD = $(shell jemalloc-config --libdir)
#JEMALLOC = -L$(JEMALLOCLD) -ljemalloc 
#endif

DEBUG = 0
PROFILE = 0
PROD = 0

ifeq ($(DEBUG), 0)
OPT = -O3 -DNDEBUG
else
OPT = -O0 -g
endif

ifneq ($(PROFILE), 0)
PROF = -pg
endif

ifeq ($(PROD), 1)
OPT += -DPROD
endif


CXXFLAGS = $(OPT) $(PROF) -mcx16 -std=c++17 -march=native -I .
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


all: $(TARGETS)

%: %.cpp $(REQUIRE)
	$(CXX) -DSTATS $(CFLAGS) -I $(INCLUDE) -o $@ $< $(LFLAGS)

clean:
	rm -f $(TARGETS)

