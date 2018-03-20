//
//  main.c
//  C_start
//
//  Created by Xiufeng Yang on 3/15/18.
//  Copyright © 2018 Xiufeng Yang. All rights reserved.
//

#include <stdio.h>
//#include <stdlib.h>
//#include <time.h>
#include <math.h>
//#include <fcntl.h>
//#include <time.h>
#define input_num 16673
#define input_dim 256
#define output_dim  1
#define hid_dim 256


#define rando() ((double)rand()/((double)RAND_MAX+1))




__global__ void z1_a1(int* d_x, double* d_W1, double* d_z1, double* d_a1, int m)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    double sum=0.0;
    
    for (int n = 0; n < 256; n++)
    {
        sum = sum+ d_x[m*256+n]*d_W1[n*256+index];
    }
    d_z1[m*256+index] = sum;
    d_a1[m*256+index] =1.0/(1.0+std::exp(-sum));
}

__device__ void z1_device(int* d_x, double* d_W1, double* d_z1, double* d_a1, int m)
{
    
    z1_a1 <<< 16, 16 >>> (d_x, d_W1,  d_z1,  d_a1,  m);
    cudaDeviceSynchronize();
    

}


__global__ void z2_a2(double* d_z2, double* d_a2, double* d_z1, double* d_a1, double* d_W2, int m)
{
    //int index = threadIdx.x;
    double sum1=0.0;
    
    for (int n = 0; n < 256; n++)
    {
    sum1 +=d_a1[m*256+n]*d_W2[n];
    }
    d_z2[m] = sum1;
    d_a2[m] =1.0/(1.0+std::exp(-sum1));
    
    
    
}

__device__ void z2_device(double* d_z2, double* d_a2, double* d_z1, double* d_a1, double* d_W2, int m)
{
    z2_a2 <<< 1, 1 >>> (d_z2,  d_a2, d_z1, d_a1,  d_W2, m);
    cudaDeviceSynchronize();
}


__global__ void W2_de(double* d_a2, double* d_Y, double* d_a1, double* d_W2, int m)
{
    // backpropagation of W2
    int index = threadIdx.x;
    double gr1=0.0;
    double lr=0.1;

 
            
    gr1=(d_a2[m]-d_Y[m])*(d_a2[m]*(1.0-d_a2[m]))*d_a1[index];
    
    d_W2[index]=d_W2[index]-lr*gr1;
            
        
    
    
}

__device__ void W2_gradient(double* d_a2, double* d_Y, double* d_a1, double* d_W2, int m)
{
    W2_de <<<1, 256>>> (d_a2, d_Y, d_a1, d_W2, m);
    cudaDeviceSynchronize();
}

__global__ void W1_de(int* d_x, double* d_a2, double* d_Y, double* d_a1, double* d_W2, double* d_W1, int m )
{
    // calculate gradident of w1
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    double gr2=0.0;
    
        for (int h=0; h<256; h++)
        {
          
            gr2=(d_a2[m]-d_Y[m])*(d_a2[m]*(1.0-d_a2[m]))*d_W2[h]*(d_a1[m*256+h]*(1.0-d_a1[m*256+h]))*d_x[m*256+index];
            
            d_W1[index*256+h]=d_W1[index*256+h]-0.1*gr2;
        }
    

    
}

__device__ void W1_gradient(int* d_x, double* d_a2, double* d_Y, double* d_a1, double* d_W2, double* d_W1, int m)
{
    W1_de <<<16 , 16>>> (d_x, d_a2, d_Y, d_a1, d_W2,d_W1, m);
    cudaDeviceSynchronize();
}



__global__ void training(int* d_x, double* d_Y, 
double* d_W1, double* d_z1, double* d_a1, double* d_W2, double* d_z2, double* d_a2, int* d_randput)
{ printf("shit");
    //double T_loss=0.0, Loss=0.0;
    for (int e = 0; e < 50; ++e)
    {
        double T_loss=0.0, Loss=0.0;

        //printf("%f \n", T_loss);

        for (int p = 0; p < 16673; p++)
        {
            //Loss = 0.5*(d_Y[0]-d_a2[0])*(d_Y[0]-d_a2[0]);
            //T_loss =T_loss+Loss;
            //printf("%f \n", T_loss);
            int m=d_randput[p];
            z1_device(d_x,  d_W1,  d_z1, d_a1,  m);
            z2_device( d_z2, d_a2,  d_z1,  d_a1, d_W2,m);
            W2_gradient( d_a2,  d_Y, d_a1,  d_W2, m);
            W1_gradient(d_x, d_a2, d_Y, d_a1, d_W2,d_W1, m);
            Loss = 0.5*(d_Y[m]-d_a2[m])*(d_Y[m]-d_a2[m]);
            
            T_loss =T_loss+Loss;


            
        }
     printf("loss: %f \n", (1.0/16673.0)*T_loss);

        
    }

}




int main(void)


{
    int i, j;
    //double sum1=0.0, sum2=0.0, gr1=0.0, gr2=0.0;
    //double Loss=0.0, T_loss=0.0, lr = 0.1;
    
    /* allocate memory size for training data X, Y and W1, W2, z1, z2, a1, a2 on GPU */
    //const long signed int insize = input_num*input_dim*sizeof(int);
    //const long signed int outsize = input_num*output_dim*sizeof(double);

    
      float gpu_elapsed_time_ms, cpu_elapsed_time_ms;
    int* randput=(int*)malloc(input_num*sizeof(int));
    // some events to count the execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
// start to count execution time of GPU version
    cudaEventRecord(start, 0);
    
    double* W1=(double*)malloc(input_dim*hid_dim*sizeof(double));
    
    double* W2=(double*)malloc(hid_dim*output_dim*sizeof(double));
    
    double* z1=(double*)malloc(input_num*hid_dim*sizeof(double));
    
    double* a1=(double*)malloc(input_num*hid_dim*sizeof(double));
    
    double* z2=(double*)malloc(input_num*output_dim*sizeof(double));
    
    double* a2=(double*)malloc(input_num*output_dim*sizeof(double));
 
    int* X=(int*)malloc(input_num*input_dim*sizeof(int));
    
    double* Y=(double*)malloc(input_num*output_dim*sizeof(double));
    double* h_a2=(double*)malloc(input_num*output_dim*sizeof(double));

    
    
    /* initialize training data, W1 and W2*/
    
    FILE *file;
    file=fopen("/home/yang/X.txt", "r");
    
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
    file1=fopen("/home/yang/Y.txt", "r");
    for(i = 0; i < input_num; i++)
    {
        for(j = 0; j < output_dim; j++)
        {
            //Use lf format specifier, %c is for character
            if (!fscanf(file1, "%lf", &Y[i*output_dim+j]))
                break;
            // mat[i][j] -= '0';
            //printf("%lf\n",Y[i*output_dim+j]); //Use lf format specifier, \n is for new line
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
    for(int k=input_num-1; k>=0; k--)
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
    double *d_Y;
    double *d_W1, *d_W2, *d_z1, *d_a1, *d_z2, *d_a2;
    int *d_randput;
    //const long signed int insize = input_num*input_dim*sizeof(int);
    //const long signed int outsize = input_num*output_dim*sizeof(double);

    cudaMalloc(&d_x, input_num*input_dim*sizeof(int));
    cudaMalloc(&d_Y,input_num*output_dim*sizeof(double));
    cudaMalloc(&d_W1, input_dim*hid_dim*sizeof(double));
    cudaMalloc(&d_z1, input_num*hid_dim*sizeof(double));
    cudaMalloc(&d_a1, input_num*hid_dim*sizeof(double));
    cudaMalloc(&d_W2, hid_dim*output_dim*sizeof(double));
    cudaMalloc(&d_z2, input_num*output_dim*sizeof(double));
    cudaMalloc(&d_a2, input_num*output_dim*sizeof(double));
    cudaMalloc(&d_randput, input_num*sizeof(int));
    
    
    /* copy memory allocated for matrices from host to device*/
    cudaMemcpy(d_x, X, input_num*input_dim*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y, Y, input_num*output_dim*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(d_W1, W1, input_dim*hid_dim*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(d_z1, z1, input_num*input_dim*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(d_a1, a1, input_num*input_dim*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(d_W2, W2, hid_dim*output_dim*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(d_z2, z2, input_num*output_dim*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(d_a2, a2, input_num*output_dim*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(d_randput, randput, input_num*sizeof(int),cudaMemcpyHostToDevice);



    
    /*start the kernel to perform stochastic gradient descent*/
    
    
    training <<< 1, 1 >>> (d_x, d_Y, d_W1, d_z1, d_a1, d_W2, d_z2, d_a2, d_randput);

    cudaMemcpy(h_a2, d_a2,input_num*output_dim*sizeof(double), cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();
    // time counting terminate
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // compute time elapse on GPU computing
    cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
    printf("Time elapsed training  on GPU: %f ms.\n\n",gpu_elapsed_time_ms);
    for (int i = 0; i < 100; i++)
{
 
printf("Predicted : %f True : %f  \n",  h_a2[i],  Y[i]);
	}

    
    cudaFree(d_x);
    cudaFree(d_Y);
    cudaFree(d_W1);
    cudaFree(d_W2);
    cudaFree(d_z1);
    cudaFree(d_z2);
    cudaFree(d_a1);
    cudaFree(d_a2);
    cudaFree(d_randput);
     
    free(W1);
    free(W2);
    free(z1);
    free(a1);
    free(a2);
    //free(X);
    //free(Y);
    free(randput);
    
    
    
    return 1 ;
    
}
