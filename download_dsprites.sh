#!/bin/bash

mkdir temp; cd temp
git clone https://github.com/deepmind/dsprites-dataset.git
mv dsprites-dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz ../data/
cd ..; rm -rf temp
