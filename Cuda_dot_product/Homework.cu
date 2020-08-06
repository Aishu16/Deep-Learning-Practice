#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <cuda.h>
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
	int CUDA_DEVICE = atoi(argv[5]);
	int THREADS = atoi(argv[6]);
	
	printf("Rows= %d,Cols = %d,CUDA_DEVICE= %d, THREADS =%d \n",rows,cols,CUDA_DEVICE,THREADS);
	cudaError err = cudaSetDevice(CUDA_DEVICE);

	if(err != cudaSuccess) { printf("Error setting CUDA DEVICE\n"); exit(EXIT_FAILURE); }

	/*Host variable declaration */

	int BLOCKS;
	struct timeval starttime, endtime;
	float* host_results = (float*) malloc(rows * sizeof(float));
	clock_t start, end;
	float seconds = 0;
	unsigned int jobs; 
	unsigned long i;

	/*Kernel variable declaration */
	int vrow =1;
	float *results;
	float *v;
	float  *data;
	//size_t len = 0;
	float arr[rows][cols];
	float var ;
	

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

        printf("time to read data = %f\n", seconds);

	/* allocate the Memory in the GPU for data */	   
        gettimeofday(&starttime, NULL);
	err = cudaMalloc((float**) &data, (size_t) size * (size_t) sizeof(float));
	if(err != cudaSuccess) { printf("Error mallocing data on GPU device\n"); }
        gettimeofday(&endtime, NULL); seconds=((double)endtime.tv_sec+(double)endtime.tv_usec/1000000)-((double)starttime.tv_sec+(double)starttime.tv_usec/1000000);

        gettimeofday(&starttime, NULL);


	/* allocate the memory in the GPU for v */
        err = cudaMalloc((float**) &v, sizeV * sizeof(float));
       if(err != cudaSuccess) { printf("Error mallocing data on GPU device\n"); }
        gettimeofday(&endtime, NULL); seconds=((double)endtime.tv_sec+(double)endtime.tv_usec/1000000)-((double)starttime.tv_sec+(double)starttime.tv_usec/1000000);
        


        gettimeofday(&starttime, NULL);
	
	/* allocate memory to result*/
	err = cudaMalloc((float**) &results, rows * sizeof(float) );
	if(err != cudaSuccess) { printf("Error mallocing results on GPU device\n"); }
        gettimeofday(&endtime, NULL); 
seconds=((double)endtime.tv_sec+(double)endtime.tv_usec/1000000)-((double)starttime.tv_sec+(double)starttime.tv_usec/1000000);
	

	/*Copy the data to GPU */
        gettimeofday(&starttime, NULL);
	err = cudaMemcpy(data, dataT, (size_t)size *sizeof(float), cudaMemcpyHostToDevice);
	if(err != cudaSuccess) { printf("Error copying data to GPU\n"); }
        gettimeofday(&endtime, NULL); seconds=((double)endtime.tv_sec+(double)endtime.tv_usec/1000000)-((double)starttime.tv_sec+(double)starttime.tv_usec/1000000);
	printf("time to copy  data to GPU=%f\n", seconds);

	/* Copy the v data to GPU*/
	gettimeofday(&starttime, NULL);
        err = cudaMemcpy(v, dataV, sizeV*sizeof(float), cudaMemcpyHostToDevice);
        if(err != cudaSuccess) { printf("Error copying data to GPU\n"); }
        gettimeofday(&endtime, NULL); seconds=((double)endtime.tv_sec+(double)endtime.tv_usec/1000000)-((double)starttime.tv_sec+(double)starttime.tv_usec/1000000);
        printf("time to copy v data to GPU=%f\n", seconds);

	jobs = rows;
	BLOCKS = (jobs + THREADS - 1)/THREADS;

        gettimeofday(&starttime, NULL);
	/*Calling the kernel function */
	kernel<<<BLOCKS,THREADS>>>(rows,cols,data,	v, results);
        gettimeofday(&endtime, NULL); seconds=((double)endtime.tv_sec+(double)endtime.tv_usec/1000000)-((double)starttime.tv_sec+(double)starttime.tv_usec/1000000);
	printf("time for kernel=%f\n", seconds);
		
	/*Copy the results back in host*/
	cudaMemcpy(host_results,results,rows * sizeof(float),cudaMemcpyDeviceToHost);
	
	printf("test output\n");
	printf("\n");
	
	for(int k = 0; k < jobs; k++) {
		printf("%f ", host_results[k]);
		printf("\n");
	}
	printf("\n");

	cudaFree( data );
	cudaFree( results );

	end = clock();
	seconds = (float)(end - start) / CLOCKS_PER_SEC;
	printf("time = %f\n", seconds);

	return 0;

}
