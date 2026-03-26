#!/usr/bin/env bash

set -e

CC=${CC:=clang}

clang -march=native -lm -fsanitize=address -fsanitize=undefined -g -O1 -DTEST wht.c -o wht.test.$CC
clang -march=native -lm -O2 -DNDEBUG wht.c -o wht.perf.$CC

./wht.test.$CC
./wht.perf.$CC
