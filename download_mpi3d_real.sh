#!/bin/bash

mkdir temp; cd temp
wget https://storage.googleapis.com/disentanglement_dataset/Final_Dataset/mpi3d_real.npz
python3 ../process_mpi.py
mv mpi3d_real_images.npy ../data/
cd ..; rm -rf temp
