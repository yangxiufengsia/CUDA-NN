# CUDA_NN

This is a CUDA implementation of simple artificial neural network(ANN) for regression. 

ANN architecture contains three layers: input layer, hidden layer and output layer.
input dimension: 256
hiddden dimension: 256
output  dimension: 1

# How to use?
nvcc -arch=sm_35 -rdc=true -I/usr/local/cuda-8.0/include -L/usr/local/cuda-8.0/lib64  main.cu -o birende.o  -lcudadevrt
