#!/usr/bin/env bash

if [ ! -d "data" ]; then
    wget -c http://data.neuralnoise.com/l2x-data.tar.gz
    tar xvfz l2x-data.tar.gz
    rm -f l2x-data.tar.gz
    wget -c http://data.neuralnoise.com/sst-data.tar.gz
    tar xvfz sst-data.tar.gz
    mv sst-data/*.npy data/
    rm -rf sst-data/ sst-data.tar.gz
fi
