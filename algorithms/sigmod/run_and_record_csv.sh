#!/bin/bash

# build executables
make query measure_recall

# run executables
./query -r data/contest-data-release-1m.bin data/contest-queries-release-1m.bin \
                    data/contest-queries-release-1m-output-graph.bin

./measure_recall -r data/contest-queries-release-1m-output-graph.bin \
           data/gt-1m-1m.bin data/contest-queries-release-1m.bin