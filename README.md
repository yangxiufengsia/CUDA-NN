# CUDA_NN

This is a CUDA implementation of simple artificial neural network(ANN) for regression. 

ANN architecture contains three layers: input layer, hidden layer and output layer.
input dimension: 256
hiddden dimension: 256
output  dimension: 1

# How to use?

1. For ANN on GPU, type the following command
nvcc -arch=sm_35 -rdc=true -I/usr/local/cuda-8.0/include -L/usr/local/cuda-8.0/lib64  cuda_nn.cu -o birende.o  -lcudadevrt

2. For ANN on CPU, type the following command
gcc cpu_nn.c -o test.o

# Checking speedup
After running the above two version code, you can compare the speed between CPU and GPU.
