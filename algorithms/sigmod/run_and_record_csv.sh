#!/bin/bash

# build executables
make query
make measure_recall

# run executables
./measure_recall -r data/contest-queries-release-1m-output-graph.bin \
           data/gt-1m-1m.bin data/contest-queries-release-1m.bin

./query -r data/contest-data-release-1m.bin data/contest-queries-release-1m.bin \
                    data/contest-queries-release-1m-output-graph.bin