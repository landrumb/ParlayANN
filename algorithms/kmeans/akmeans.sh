#!bin/bash
bazel build kmeans_test_run

PARANN_DIR=~/ParlayANN

$PARANN_DIR/bazel-bin/algorithms/kmeans/kmeans_test_run -k 200 -i /ssd1/anndata/bigann/base.1B.u8bin.crop_nb_1000000 -f bin -t uint8 -D fast -m 5 -two yes 