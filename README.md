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



# Training results

loss: 0.004168 
loss: 0.003697 
loss: 0.003471 
loss: 0.003288 
loss: 0.003131 
loss: 0.002994 
loss: 0.002872 
loss: 0.002763 
loss: 0.002664 
loss: 0.002573 
loss: 0.002489 
loss: 0.002412 
loss: 0.002340 
loss: 0.002274 
loss: 0.002211 
loss: 0.002153 
loss: 0.002098 
loss: 0.002047 
loss: 0.001999 
loss: 0.001953 
loss: 0.001910 
loss: 0.001869 
loss: 0.001830 
loss: 0.001793 
loss: 0.001758 
loss: 0.001725 
loss: 0.001693 
loss: 0.001662 
loss: 0.001633 
loss: 0.001605 
loss: 0.001578 
loss: 0.001552 
loss: 0.001528 
loss: 0.001504 
loss: 0.001481 
loss: 0.001459 
loss: 0.001437 
loss: 0.001417 
loss: 0.001397 
loss: 0.001378 
loss: 0.001359 
loss: 0.001342 
loss: 0.001324 
loss: 0.001307 
loss: 0.001291 
loss: 0.001275 
loss: 0.001260 
loss: 0.001245 
loss: 0.001231 
Time elapsed training  on GPU: 231026.109375 ms.

Predicted : 0.275531 True : 0.321664  
Predicted : 0.414367 True : 0.362600  
Predicted : 0.369136 True : 0.380842  
Predicted : 0.175931 True : 0.152717  
Predicted : 0.577158 True : 0.597249  
Predicted : 0.371012 True : 0.371866  
Predicted : 0.241326 True : 0.239524  
Predicted : 0.574280 True : 0.570344  
Predicted : 0.155775 True : 0.143427  
Predicted : 0.206300 True : 0.214830  
Predicted : 0.222317 True : 0.201784  
Predicted : 0.163485 True : 0.224400  
Predicted : 0.283107 True : 0.288033  
Predicted : 0.435239 True : 0.324108  
Predicted : 0.406492 True : 0.440051  
Predicted : 0.447094 True : 0.370323  
Predicted : 0.201878 True : 0.282296  
Predicted : 0.412801 True : 0.415543  
Predicted : 0.199248 True : 0.217393  
Predicted : 0.550738 True : 0.533686  
Predicted : 0.191824 True : 0.210582  
Predicted : 0.163744 True : 0.143756  
Predicted : 0.184900 True : 0.211496  
Predicted : 0.289283 True : 0.255965  
Predicted : 0.338155 True : 0.345928  
Predicted : 0.224460 True : 0.145335  
Predicted : 0.087297 True : 0.078090  
Predicted : 0.170139 True : 0.137583  
Predicted : 0.402358 True : 0.416847  
Predicted : 0.272623 True : 0.268498  
Predicted : 0.333361 True : 0.389053  
Predicted : 0.379425 True : 0.426745  
Predicted : 0.262207 True : 0.236274  
Predicted : 0.272567 True : 0.259111  
Predicted : 0.055733 True : 0.081743  
Predicted : 0.445414 True : 0.417329  
Predicted : 0.347888 True : 0.238497  
Predicted : 0.240060 True : 0.229261  
Predicted : 0.452913 True : 0.446108  
Predicted : 0.396364 True : 0.373477  
Predicted : 0.253710 True : 0.219664  
Predicted : 0.127238 True : 0.096519  
Predicted : 0.174973 True : 0.238495  
Predicted : 0.092660 True : 0.125481  
Predicted : 0.128991 True : 0.105110  
Predicted : 0.155202 True : 0.142715  
Predicted : 0.465487 True : 0.485416  
Predicted : 0.390846 True : 0.399472  
Predicted : 0.389793 True : 0.302352  
Predicted : 0.378935 True : 0.365606  
Predicted : 0.532431 True : 0.520932  
Predicted : 0.140734 True : 0.130441  
Predicted : 0.099136 True : 0.081297  
Predicted : 0.321540 True : 0.316336  
Predicted : 0.606810 True : 0.541097  
Predicted : 0.194503 True : 0.166844  
Predicted : 0.571595 True : 0.548714  
Predicted : 0.210992 True : 0.203312  
Predicted : 0.262393 True : 0.285624  
Predicted : 0.497681 True : 0.578986  
Predicted : 0.243686 True : 0.261693  
Predicted : 0.608970 True : 0.578854  
Predicted : 0.212263 True : 0.220965  
Predicted : 0.436876 True : 0.434967  
Predicted : 0.124170 True : 0.149758  
Predicted : 0.226610 True : 0.189022  
Predicted : 0.095198 True : 0.110588  
Predicted : 0.166073 True : 0.111940  
Predicted : 0.363991 True : 0.383409  
Predicted : 0.336235 True : 0.361604  
Predicted : 0.467244 True : 0.426174  
Predicted : 0.285344 True : 0.297547  
Predicted : 0.067118 True : 0.108011  
Predicted : 0.297454 True : 0.262281  
Predicted : 0.364597 True : 0.393368  
Predicted : 0.194487 True : 0.175990  
Predicted : 0.281791 True : 0.199267  
Predicted : 0.140578 True : 0.104858  
Predicted : 0.113214 True : 0.196853  
Predicted : 0.631473 True : 0.585527  
Predicted : 0.332680 True : 0.275084  
Predicted : 0.176618 True : 0.165530  
Predicted : 0.506482 True : 0.478443  
Predicted : 0.463844 True : 0.387555  
Predicted : 0.369922 True : 0.278435  
Predicted : 0.149109 True : 0.128280  
Predicted : 0.447335 True : 0.491122  
Predicted : 0.222115 True : 0.220343  
Predicted : 0.106694 True : 0.093775  
Predicted : 0.350010 True : 0.338536  
Predicted : 0.261473 True : 0.272661  
Predicted : 0.173952 True : 0.133178  
Predicted : 0.150802 True : 0.188415  
Predicted : 0.130293 True : 0.200142  
Predicted : 0.193066 True : 0.189516  
Predicted : 0.133251 True : 0.130649  
Predicted : 0.096188 True : 0.110019  
Predicted : 0.598313 True : 0.541821  
Predicted : 0.090759 True : 0.063908  
Predicted : 0.266879 True : 0.247468  
