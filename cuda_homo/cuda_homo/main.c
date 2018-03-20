//
//  main.c
//  C_start
//
//  Created by Xiufeng Yang on 3/15/18.
//  Copyright © 2018 Xiufeng Yang. All rights reserved.
//

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <fcntl.h>
#include <time.h>
#define input_num 16673
#define input_dim 256
#define output_dim  1
#define hid_dim 28


#define rando() ((double)rand()/((double)RAND_MAX+1))




__global__ void z1_a1(int* d_x, double* d_W1, double* d_z1, int m)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int sum=0;
    
    for (int n = 0; n < 256; n++)
    {
        sum = sum+ X[m*256+n]*W1[index*256+n];
    }
    d_z1[index] = sum;
    d_a1[index] =1.0/(1.0+exp(-d_z1[index]));
}

__device__ void z1_device(int* d_x, double* d_W1, double* d_z1, int m)
{
    
    z1_a1 <<< 16, 16 >>> (int* d_x, double* d_W1, double* d_z1, int m);
    cudaDeviceSynchronize();
    

}


__global__ void z2(double* d_z2, double* d_a2, double* d_z1, double* d_a1, double* d_W2)
{
    int index = threadIdx.x;
    int sum1=0;
    
    for (int n = 0; n < 256; n++)
    {
        sum1 = sum1+ a1[n]*W1[n];
    }
    d_z2[index] = sum1;
    d_a2[index] =1.0/(1.0+exp(-d_z2[index]));
    
    
    
}

__device__ void z2_device(double* d_z2, double* d_a2, double* d_z1, double* d_a1, double* d_W2)
{
    z2_a2 <<< 1, 1 >>> (double* d_z2, double* d_a2, double* d_z1, double* d_a1, double* d_W2);
    cudaDeviceSynchronize();
}


__global__ void W2_de(double* d_a2, double* d_Y, double* d_a1, double* d_W2)
{
    // backpropagation of W2
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    double gr1=0.0;

 
            
    gr1=(d_a2[0]-d_Y[0])*(d_a2[0]*(1.0-d_a2[0]))*d_a1[index];
    
    d_W2[index]=d_W2[index]-lr*gr1;
            
        }
    }
    
}

__device__ void W2_gradient(double* d_a2, double* d_Y, double* d_a1, double* d_W2)
{
    W2_de <<< 16, 16>>> (double* d_a2, double* d_Y, double* d_a1, double* d_W2);
    cudaDeviceSynchronize();
}

__global__ void W1_de()
{
    // calculate gradident of w1
    for (n=0; n < input_dim; n++)
    {
        for (h=0; h<hid_dim; h++)
        {
          
            gr2=(a2[0]-Y[0])*(a2[0]*(1.0-a2[0]))*W2[index]*(a1[index]*(1.0-a1[index]))*X[m*256+index];
            
            W1[n*hid_dim+h]=W1[n*hid_dim+h]-lr*gr2;
        }
    }

    
}

__device__ void W1_gradient()
{
    W1_de <<< 16, 16>>> (double* d_a2, double* d_Y, double* d_a1, double* d_W2);
    cudaDeviceSynchronize();
}



__global__ void kFit(int* d_x, double* d_Y, double* d_W1, double*　d_z1, double* d_a1, double* d_W2, double* d_z2, double* d_a2, int* d_randput, int tr_num, int tr_dim, int l1_dim int l2_dim)
{
    double T_loss=0.0, Loss=0.0;
    for (int epoch = 0; epoch < 50; ++epoch)
    {
        
        for (int p = 0; p < tr_num; p++)
        {
            Loss = 0.5*(d_Y[0]-d_a2[0])*(Y[0]-a2[0]);
            T_loss =T_loss+Loss;
            int m=d_randput[p]
            z1_device(int tr_dim, int l1_dim, int* d_x, double* d_W1, double* d_z1, int m);
            z2_device(double* d_z2, double* d_a2, double* d_z1, double* d_a1, double* d_W2);
            W2_gradient();
            W1_gradient();
            
        }
        
    }
}




int main(int argc, const char * argv[])

{
    int i, j, m, n, h, o, k, p, epoch;
    double sum1=0.0, sum2=0.0, gr1=0.0, gr2=0.0;
    double Loss=0.0, T_loss=0.0, lr = 0.1;
    
    /* allocate memory size for training data X, Y and W1, W2, z1, z2, a1, a2 on GPU */
    int* randput=malloc(input_num*sizeof(int));
    
    double* W1=malloc(input_dim*hid_dim*sizeof(double));
    
    double* W2=malloc(hid_dim*output_dim*sizeof(double));
    
    double* z1=malloc(hid_dim*sizeof(double));
    
    double* a1=malloc(hid_dim*sizeof(double));
    
    double* z2=malloc(output_dim*sizeof(double));
    
    double* a2=malloc(output_dim*sizeof(double));
    
    
    int* X=malloc(input_num*input_dim*sizeof(int));
    
    double* Y=malloc(input_num*output_dim*sizeof(double));
    
    
    /* initialize training data, W1 and W2*/
    
    FILE *file;
    file=fopen("/Users/yang/Downloads/X.txt", "r");
    
    for(i = 0; i < input_num; i++)
    {
        for(j = 0; j < input_dim; j++)
        {
            //Use lf format specifier, %c is for character
            if (!fscanf(file, "%d", &X[i*input_dim+j]))
                break;
            // mat[i][j] -= '0';
            //printf("%d\n",X[i*input_dim+j]); //Use lf format specifier, \n is for new line
        }
        
    }
    
    FILE *file1;
    file1=fopen("/Users/yang/Downloads/Y.txt", "r");
    for(i = 0; i < input_num; i++)
    {
        for(j = 0; j < output_dim; j++)
        {
            //Use lf format specifier, %c is for character
            if (!fscanf(file1, "%lf", &Y[i*output_dim+j]))
                break;
            // mat[i][j] -= '0';
            //printf("%lf\n",Y[i][j]); //Use lf format specifier, \n is for new line
        }
        
    }
    
    fclose(file);
    fclose(file1);
    
    
    for (i=0; i<input_num; i++)
    {
        randput[i]=i;
    }
    
    
    for( i= 0 ; i <input_dim; i++ )
    {    /* initialize Weight W1*/
        for( j = 0 ; j <hid_dim; j++ )
        {
            W1[i*hid_dim+j] =2.0*(rando()-0.5)*0.5;
        }
    }
    
    for( i= 0 ; i <hid_dim; i++ )
    {    /* initialize Weight W1*/
        for( j = 0 ; j <output_dim; j++ )
        {
            W2[i*output_dim+j] =2.0*(rando()-0.5)*0.5;
        }
    }
    
    // randomize the order of training data
    srand (time(NULL));
    for(k=input_num-1; k>=0; k--)
    {
        j=rand() % (k+1);
        int temp=randput[k];
        randput[k]=randput[j];
        randput[j]=temp;
        
    }
    //printf("%f \n",(0.5/16673)*T_loss);
    //Loss=0.0;
    //T_loss=0.0;
    
    /* Allocate matrices d_x, d_y, d_W1, d_W2, d_z1, d_a1, d_z2, d_a2 on device*/
    
    int *d_x;
    double *d_y;
    double *d_W1, *d_W2, *d_z1, *d_a1, *d_z2, *d_a2;
    int *d_randput;
    
    cudaMalloc(&d_x, X, input_num*input_dim*sizeof(int));
    cudaMalloc(&d_y,Y, input_num*output_dim*sizeof(double));
    cudaMalloc(&d_W1,W1, input_dim*hid_dim*sizeof(double));
    cudaMalloc(&d_z1,z1, hid_dim*sizeof(double));
    cudaMalloc(&d_a1,a1, hid_dim*sizeof(double));
    cudaMalloc(&d_W2,W2, hid_dim*output_dim*sizeof(double));
    cudaMalloc(&d_z2,z2, output_dim*sizeof(double));
    cudaMalloc(&d_a2,a2, output_dim*sizeof(double));
    cudaMalloc(&d_randput, randput, input_num*sizeof(int));
    
    
    /* copy memory allocated for matrices from host to device*/
    cudaMemcpy(d_x, X, input_num*input_dim*sizeof(int)),cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, Y, input_num*output_dim*sizeof(double)),cudaMemcpyHostToDevice);
    cudaMemcpy(d_W1, W1, input_dim*hid_dim*sizeof(double)),cudaMemcpyHostToDevice);
    cudaMemcpy(d_z1, z1, input_num*input_dim*sizeof(double)),cudaMemcpyHostToDevice);
    cudaMemcpy(d_a1, a1, input_num*input_dim*sizeof(double)),cudaMemcpyHostToDevice);
    cudaMemcpy(d_W2, W2, hid_dim*output_dim*sizeof(double)),cudaMemcpyHostToDevice);
    cudaMemcpy(d_z2, z2, input_num*output_dim*sizeof(double)),cudaMemcpyHostToDevice);
    cudaMemcpy(d_a2, a2, input_num*output_dim*sizeof(double)),cudaMemcpyHostToDevice);
    cudaMemcpy(d_randput, randput, input_num*sizeof(int)),cudaMemcpyHostToDevice);



    
    /*start the kernel to perform stochastic gradient descent*/
    
    
    kFit <<< 1, 1 >>> (	d_x, d_y, d_W1, d_z1, d_a1, d_W2, d_z2, d_a2, d_randput, input_num, input_dim, hid_dim output_dim);

    cudaMemcpy(d_W1, d_W2, d_y, cudaMemcpyDeviceToHost);
    
    cudaFree(d_X);
    cudaFree(d_Y);
    cudaFree(d_W1);
    cudaFree(d_W2);
    cudaFree(d_z1);
    cudaFree(d_z2);
    cudaFree(d_a1);
    cudaFree(d_a2);
    cudaFree(d_randput);
    
    
    
    return 1 ;
    
}
