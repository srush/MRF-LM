#!/bin/bash

cd liblbfgs-1.10/
./configure
make
cd ..
make
make mrflm_test
