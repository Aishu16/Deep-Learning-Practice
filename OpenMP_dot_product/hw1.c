#include <stdio.h>
//#include <iostream>
#include <stdlib.h>
#include <string.h>
//#include <cuda.h>
#include <time.h>
#include <sys/time.h>
//#include <cuda_runtime_api.h>
#include "Chi2.h"

int main(int argc ,char* argv[]) {

	FILE *vp;	
	FILE *fp;
	size_t size;
	  
	/* Initialize rows, cols, ncases, ncontrols from the user */
	unsigned int rows=atoi(argv[3]);
	unsigned int cols=atoi(argv[4]);
	int nprocs = atoi(argv[5]);
	
	printf("Rows = %d,Cols = %d,nprocs =%d \n",rows,cols,nprocs);

	/*Host variable declaration */
	int BLOCKS;
	float* host_results = (float*) malloc(rows * sizeof(float)); 
	struct timeval starttime, endtime;
	clock_t start, end;
	float seconds = 0;
	unsigned int jobs; 
	unsigned long i;

	/*Kernel variable declaration */
	float  *dev_dataT;
	float *dev_dataV;
	float *results;
        //size_t len = 0;
	float arr[rows][cols];
	float var ;
	int vrow =1;

	start = clock();

	/* Validation to check if the data file is readable */
	fp = fopen(argv[1], "r");
	vp = fopen(argv[2],"r");
	
	if (fp == NULL) {
    		printf("Cannot Open the File");
		return 0;
	}
	if (vp == NULL){
		printf("cannot open the file");
	}
	size = (size_t)((size_t)rows * (size_t)cols);
	size_t sizeV = 0;
	sizeV = (size_t)((size_t)vrow*(size_t)cols);

	/*printf("Size of the data = %lu\n",size);*/

	fflush(stdout);

	float *dataT = (float*)malloc((size)*sizeof(float));
	float *dataV = (float*)malloc((sizeV) * sizeof(float));

	if(dataT == NULL) {
	        printf("ERROR: Memory for data not allocated.\n");
	}
	
	if(dataV == NULL){
		printf("ERROR: Memory for data not allocated. \n");
	}
        gettimeofday(&starttime, NULL);
	int j = 0;
        /* Transfer the Data from the file to CPU Memory */
        for (i =0; i< rows;i++){
		for(j=0; j<cols ; j++){
			fscanf(fp,"%f",&var);
                        arr[i][j]=var;
		}
	}
	for (i =0;i<cols;i++){
		for(j= 0; j<rows; j++){
			dataT[rows*i+j]= arr[j][i];
		}
	}		

		for (j=0;j<cols;j++){
			fscanf(vp,"%f",&dataV[j]);
		}
   
	fclose(fp);
	fclose(vp);

        fflush(stdout);

        gettimeofday(&endtime, NULL);
        seconds+=((double)endtime.tv_sec+(double)endtime.tv_usec/1000000)-((double)starttime.tv_sec+(double)starttime.tv_usec/1000000);

        printf(" timw = %f\n", seconds);

	/* allocate the Memory in the GPU for data */	   
        /* gettimeofday(&starttime, NULL);
	err = cudaMalloc((float**) &dev_dataT, (size_t) size * (size_t) sizeof(float));
	if(err != cudaSuccess) { printf("Error mallocing data on GPU device\n"); }
        gettimeofday(&endtime, NULL); seconds=((double)endtime.tv_sec+(double)endtime.tv_usec/1000000)-((double)starttime.tv_sec+(double)starttime.tv_usec/1000000);
	printf("time for cudamalloc for data =%f\n", seconds);

        gettimeofday(&starttime, NULL);


	/* allocate the memory in the GPU for vector */
       /* err = cudaMalloc((float**) &dev_dataV, sizeV * sizeof(float));
       if(err != cudaSuccess) { printf("Error mallocing data on GPU device\n"); }
        gettimeofday(&endtime, NULL); seconds=((double)endtime.tv_sec+(double)endtime.tv_usec/1000000)-((double)starttime.tv_sec+(double)starttime.tv_usec/1000000);
        printf("time for cudamalloc for vector =%f\n", seconds);

        gettimeofday(&starttime, NULL);
	
	/* allocate memory to result*/
/*	err = cudaMalloc((float**) &results, rows * sizeof(float) );
	if(err != cudaSuccess) { printf("Error mallocing results on GPU device\n"); }
        gettimeofday(&endtime, NULL); 
seconds=((double)endtime.tv_sec+(double)endtime.tv_usec/1000000)-((double)starttime.tv_sec+(double)starttime.tv_usec/1000000);
	printf("time for cudamalloc for result =%f\n", seconds);

	/*Copy the data to GPU */
  /*      gettimeofday(&starttime, NULL);
	err = cudaMemcpy(dev_dataT, dataT, (size_t)size *sizeof(float), cudaMemcpyHostToDevice);
	if(err != cudaSuccess) { printf("Error copying data to GPU\n"); }
        gettimeofday(&endtime, NULL); seconds=((double)endtime.tv_sec+(double)endtime.tv_usec/1000000)-((double)starttime.tv_sec+(double)starttime.tv_usec/1000000);
	printf("time to copy  data to GPU=%f\n", seconds);

	/* Copy the VECTOR data to GPU*/
/*	gettimeofday(&starttime, NULL);
        err = cudaMemcpy(dev_dataV, dataV, sizeV*sizeof(float), cudaMemcpyHostToDevice);
        if(err != cudaSuccess) { printf("Error copying data to GPU\n"); }
        gettimeofday(&endtime, NULL); seconds=((double)endtime.tv_sec+(double)endtime.tv_usec/1000000)-((double)starttime.tv_sec+(double)starttime.tv_usec/1000000);
        printf("time to copy vector data to GPU=%f\n", seconds); */

	jobs = (unsigned int)((rows +nprocs -1)/nprocs);
	//BLOCKS = (jobs + THREADS - 1)/THREADS;

        gettimeofday(&starttime, NULL);
        
	/*jobs */
        printf("%d", jobs);

        #pragma omp parallel num_threads(nprocs)
        kernel(rows,cols,dataT,dataV,host_results,jobs);
        
	gettimeofday(&endtime, NULL); seconds=((double)endtime.tv_sec+(double)endtime.tv_usec/1000000)-((double)starttime.tv_sec+(double)starttime.tv_usec/1000000);
	printf("time for kernel=%f\n", seconds);
	printf("oputput\n");

	printf("\n");
	int k;
	for(k = 0; k < rows; k++) {
		printf("%f ", host_results[k]);
		//printf("\n");
	}
	printf("\n");

	end = clock();
	seconds = (float)(end - start) / CLOCKS_PER_SEC;
	printf("Total time= %f\n", seconds);

	return 0;

}

