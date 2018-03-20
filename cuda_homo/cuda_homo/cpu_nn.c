//
//  main.c
//  C_start
//
//  Created by 楊しゅうほう on 3/7/18.
//  Copyright © 2018 楊しゅうほう. All rights reserved.
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
#define hid_dim 256


#define rando() ((double)rand()/((double)RAND_MAX+1))

int main(int argc, const char * argv[])

{
    int i, j, m, n, h, o, k, p, epoch;
    double sum1=0.0, sum2=0.0, gr1=0.0, gr2=0.0;
    double Loss=0.0, T_loss=0.0, lr = 0.001;

    /* allocate memory size for W1, W2, z1, z2, a1, a2*/
    int* randput=malloc(input_num*sizeof(int));
    
    double** W1=malloc(input_dim*sizeof(double*));
    for(i=0;i<input_dim;++i)
    {
        W1[i]=malloc(hid_dim*sizeof(double));
    }
    double** W2=malloc(hid_dim*sizeof(double*));
    for(i=0;i<hid_dim;++i)
    {
        W2[i]=malloc(output_dim*sizeof(double));
    }
    
    double** z1=malloc(input_num*sizeof(double*));
    for(i=0;i<input_num;++i)
    {
        z1[i]=malloc(hid_dim*sizeof(double));
    }
    double** a1=malloc(input_num*sizeof(double*));
    for(i=0;i<input_num;++i)
    {
        a1[i]=malloc(hid_dim*sizeof(double));
    }
    
    
    double** z2=malloc(input_num*sizeof(double*));
    for(i=0;i<input_num;++i)
    {
        z2[i]=malloc(output_dim*sizeof(double));
    }
    double** a2=malloc(input_num*sizeof(double*));
    for(i=0;i<input_num;++i)
    {
        a2[i]=malloc(output_dim*sizeof(double));
    }
    
    
    
    /*matrix*/
    /*Use double , you have floating numbers not int*/
    
    int** X=malloc(input_num*sizeof(int*));
    for(i=0;i<input_num;++i)
    {
        X[i]=malloc(input_dim*sizeof(int));
    }
    
    double** Y=malloc(input_num*sizeof(double*));
    for(i=0;i<input_num;++i)
    {
     Y[i]=malloc(output_dim*sizeof(double));
    }
    
    
    FILE *file;
    file=fopen("/Users/yang/Downloads/X.txt", "r");
    
    for(i = 0; i < input_num; i++)
    {
        for(j = 0; j < input_dim; j++)
        {
            //Use lf format specifier, %c is for character
            if (!fscanf(file, "%d", &X[i][j]))
                break;
            // mat[i][j] -= '0';
            //printf("%d\n",X[i][j]); //Use lf format specifier, \n is for new line
        }
        
    }
    
    FILE *file1;
    file1=fopen("/Users/yang/Downloads/Y.txt", "r");
    for(i = 0; i < input_num; i++)
    {
        for(j = 0; j < output_dim; j++)
        {
            //Use lf format specifier, %c is for character
            if (!fscanf(file1, "%lf", &Y[i][j]))
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
            W1[i][j] =2.0*(rando()-0.5)*0.5;
        }
    }
    
    for( i= 0 ; i <hid_dim; i++ )
    {    /* initialize Weight W1*/
        for( j = 0 ; j <output_dim; j++ )
        {
            W2[i][j] =2.0*(rando()-0.5)*0.5;
        }
    }
    
    clock_t begin = clock();
    
    
    /* Stochastic gradient descent*/
    
    for (epoch=0 ; epoch<200; epoch++)
    {
        // randomize the order of training data
        srand (time(NULL));
        for(k=input_num-1; k>=0; k--)
        {
            j=rand() % (k+1);
            int temp=randput[k];
            randput[k]=randput[j];
            randput[j]=temp;
            
        }
        
        //printf("%f \n",0.1*T_loss);
        //printf("%f \n",(0.5/16673)*T_loss);
        printf("%f \n",(0.5/16673)*T_loss);
        Loss=0.0;
        T_loss=0.0;
        
        for (p = 0; p < input_num; p++)
        {
            m=randput[p];
            //printf("%d \n", m);
            
            // calculate z1 and a1
            for (h = 0; h < hid_dim; h++)
            {
                for (n = 0; n < input_dim; n++)
                {
                    sum1 = sum1 + X[m][n]*W1[n][h];
                    //printf("%f \n",sum1);
                }
                z1[m][h] = sum1;
                //printf("%f \n",z1[m][h]);
                a1[m][h] =1.0/(1.0+exp(-z1[m][h]));
                //printf("%f \n",a1[m][h]);
                sum1=0.0;
                //printf("%f \n",a1[m][h]);
                
                
            }
            
            for (o = 0; o < output_dim; o++)
            {
                for (h = 0; h < hid_dim; h++)
                {
                    sum2 = sum2 + a1[m][h]*W2[h][o];
                    //printf("%f \n",a1[m][h]);
                    //printf("%f \n",W2[h][o]);
                }
                
                z2[m][o] = sum2;
                a2[m][o] =1.0/(1.0+exp(-z2[m][o]));
                //a2[m][o]=z2[m][o];
                //printf("%f \n",z2[m][o]);
                //printf("%f \n",a2[m][o]);
                Loss = 0.5*(Y[m][o]-a2[m][o])*(Y[m][o]-a2[m][o]);
                T_loss =T_loss+Loss;
                //printf("%f \n",(0.5/16673)*T_loss);
                
                sum2 = 0.0;
            }
            
            
            
            
            // backpropagation of W2
            for (h=0; h < hid_dim; h++)
            {
                
                for (o=0; o<output_dim; o++)
                {
                    
                    gr1=(a2[m][o]-Y[m][o])*(a2[m][o]*(1.0-a2[m][o]))*a1[m][h];
                    //gr1=(Y[m][o]-a2[m][o])*a1[m][h];
                    //printf("%f \n",gr1);
                    W2[h][o]=W2[h][o]-lr*gr1;
                    
                }
                
            }
            
            // calculate gradident of w1
            for (n=0; n < input_dim; n++)
            {
                for (h=0; h<hid_dim; h++)
                {
                    gr2=0.0;
                    for (o=0; o<output_dim; o++)
                    {
                        gr2=gr2+(a2[m][o]-Y[m][o])*(a2[m][o]*(1.0-a2[m][o]))*W2[h][o]*(a1[m][h]*(1.0-a1[m][h]))*X[m][n];
                        //printf("%f \n",gr2);
                        //printf("%f \n",a2[m][o]);
                        
                        //gr2=gr2+(a2[m][o]-Y[m][o])*W2[h][o]*(a1[m][h]*(1.0-a1[m][h]))*X[m][n];
                    }
                    //printf("%f \n",gr2);
                    
                    
                    //printf("%d \n",X[m][n]);
                    
                    W1[n][h]=W1[n][h]-lr*gr2;
                }
            }
            
        }
        
    }

    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("%f \n", time_spent);
    

    
    return 1 ;
    
}
