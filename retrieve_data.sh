#!/bin/bash
set -o errexit
set -o pipefail
set -o nounset

# Download corpora
VIZML_DATA_URL="http://vizml-repository.s3.amazonaws.com"
wget --cut-dirs=0 -e robots=off -P features "$VIZML_DATA_URL/features.tar.gz"
# wget --cut-dirs=0 -e robots=off -P data "$VIZML_DATA_URL/plotly_subset_1k.tar.gz"
wget --cut-dirs=0 -e robots=off -P data "$VIZML_DATA_URL/plotly_subset_100k.tar.gz"
# wget --cut-dirs=0 -e robots=off -P data "$VIZML_DATA_URL/plotly_full.tar.gz"

# Unzip
tar zxvf data/*.tar.gz -C data
tar zxvf features/*.tar.gz -C features
