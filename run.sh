#!/usr/bin/env bash

set -e

clang -march=native -lm -fsanitize=address -fsanitize=undefined -g -O1 -DTEST wht.c -o wht.test
clang -march=native -lm -O2 wht.c -o wht.perf

./wht.test
./wht.perf
