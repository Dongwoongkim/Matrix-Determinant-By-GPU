
#include <malloc.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

void printMatrix(double *matrix, int length); //

__device__ void MinorMat(double *M, double* Minor, int size, int row, int col); //
__device__ double cofactor(double *M, double * Minor, int size, int i, int j); //
__global__ void CofactorMat(double *M, double *N, int size);
__global__ void sumRows(double *matrix, int i, int size); //
__global__ void getDeterminant(double *matrix, int size); //
__global__ void Trans(double* M, double* P, int width); // 
__global__ void Inverse(double *M, double *P, double res); //

int main() 
{
    for(int size = 2; size <= 10; size++)
    {
    
    double *dev_X, *dev_Y, *dev_N, *dev_cfMat, *dev_M, *dev_origin;
    double *M, *tmpM, *N;

    int Matrixsize = size * size;
    int Buffersize = Matrixsize * sizeof(double);

    M = (double*)malloc(Buffersize);
    N = (double*)malloc(Buffersize);
    tmpM = (double*)malloc(Buffersize);

    cudaMalloc(&dev_M, Buffersize);  //allocating memory on gpu device
    cudaMalloc(&dev_origin,Buffersize);
    cudaMalloc(&dev_X,Buffersize);
    cudaMalloc(&dev_Y,Buffersize);
    cudaMalloc(&dev_N,Buffersize);
    cudaMalloc(&dev_cfMat,Buffersize);

    for(int i=0;i<size*size;i++)
    {
        tmpM[i] = rand()%4;
        M[i] = tmpM[i];
    }

    // 원본행렬 출력
    printf("원본행렬 M\n");
    printMatrix(M,size);
    printf("\n");

    clock_t start = clock();

    cudaMemcpy(dev_origin, tmpM, Buffersize, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_M, tmpM, Buffersize, cudaMemcpyHostToDevice);  //copying matrix

    // step1. 행렬식 구하기
    getDeterminant<<<1, 1>>> (dev_M, size);  //getting triangular matrix
    cudaDeviceSynchronize();  //waiting for all the calculations to end
    
    cudaMemcpy(tmpM, dev_M, Buffersize, cudaMemcpyDeviceToHost);  //copying matrix back
  
    double res = tmpM[0];  // actually calculating result
    
    // 대각성분 곱해서 det 추출
    for (int i = 1; i < size; i++) 
      res *= tmpM[i * size + i];

    printf("Matrix Size : %d , determinant : %f, ", size, res);
    

    // step2. 여인수행렬 구하기 
    // dev_origin -> dev_X 
    CofactorMat <<<1,size*size>>> (dev_origin, dev_X, size);
    cudaDeviceSynchronize();



    // step3. 여인수행렬T -> 수반행렬 구하기
    // dev_X -> dev_Y 
    Trans <<<size,size>>> (dev_X, dev_Y, size);
    cudaDeviceSynchronize();


    // step4. 수반행렬/det = 역행렬
    // dev_Y -> dev_N    
    Inverse<<<size,size>>>(dev_Y, dev_N,res);
    cudaDeviceSynchronize();

    
    // 역행렬 출력

    printf("\nM의 역행렬 \n");
    cudaMemcpy(N, dev_N, Buffersize, cudaMemcpyDeviceToHost);  
    printMatrix(N,size); 

    clock_t end = clock();
    printf("Time : %lfs\n", (double)(end - start) / CLOCKS_PER_SEC);

    cudaFree(dev_Y);
    cudaFree(dev_M);
    cudaFree(dev_X);
    cudaFree(dev_N);
    cudaFree(dev_cfMat);
    cudaFree(dev_origin);

    free(tmpM);
    free(M);
    free(N);

    cudaDeviceSynchronize();

    }

    return 0;   
}


void printMatrix(double *matrix, int length) 
{
    printf("\n");
    for (int i = 0; i < length; i++) {
        for (int j = 0; j < length; j++) {
            printf("%f ", matrix[i * length + j]);
        }
        printf("\n");
    }
    printf("\n");
}

__global__ void CofactorMat(double *M, double *N, int size)
{
    int tid = threadIdx.x;
    double* Minor;
    cudaMalloc( &Minor, (size-1)*(size-1)*sizeof(double));

    double d = cofactor(M, Minor, size, tid / size, tid % size);
    
    N[tid] = d;

}

__device__ double cofactor(double *M, double *Minor, int size, int i, int j)
{   
    double res;

    MinorMat(M, Minor, size, i, j);

    // M(ij)에서 i+j 가 짝수인 경우
    if((i+j)%2==0)
    {  
        getDeterminant<<<1, 1>>> (Minor, size-1);  
        cudaDeviceSynchronize();  

        res = Minor[0];  // actually calculating result

        // 대각성분 곱해서 det 추출
        for (int a = 1; a < size-1; a++)
        {
           res *= Minor[a * (size-1) + a];

        } 
        return res;
    }

    // M(ij)에서 i+j가 홀수인 경우
    else
    {
        getDeterminant<<<1, 1>>> (Minor, size-1); 
        cudaDeviceSynchronize(); 

        double res = Minor[0]; 
        
        // 대각성분 곱해서 det 추출
        for (int a = 1; a < size-1 ; a++)
        {
           res *= Minor[a * (size-1) + a];
        } 

        return -1.0*res;
    }

}

__device__ void MinorMat(double *M, double *Minor, int size, int row, int col)
{
    int cnt=0; 
    
    for(int i=0;i<size;i++)
    {   
        if(i==row) continue;

        for(int j=0;j<size;j++)
        {
            if(j==col) continue;

            else
            {
                Minor[cnt++] = M[i*size+j];
            }
        }
    }
}

__global__ void sumRows(double *matrix, int i, int size) {
    int t_id = threadIdx.x;  // getting thread id
    if (t_id > i && t_id < size) 
    { 
      // nan , inf 발생 처리
      if(matrix[i * size + i] == 0 ) matrix[i * size + i] = 0.0001;
      double factor = - (matrix[ size *t_id + i] / matrix[i * size + i]);  
      
      for (int j = i; j < size; j++)  // each thread calculating its own row
      {
        matrix[size * t_id + j] += (double) matrix[i * size + j] * factor;
      }
    }   
}

__global__ void getDeterminant(double *matrix, int size) 
{
    __syncthreads();  // putting together all the threads
    for (int i = 0; i < size; i++) 
    {
        sumRows<<<1, size>>>(matrix, i, size);  
    }
}

__global__ void Trans(double* M, double* P, int width)
{
    int tid, tx, ty;
    tx = blockDim.x * blockIdx.x + threadIdx.x;
    ty = blockDim.y * blockIdx.y + threadIdx.y;
    int DimX = gridDim.x * blockDim.x;
    tid = DimX * ty + tx;
    
    int a = tid/width;
    int b = tid%width;

    P[tid] = M[b*width+a];
}

__global__ void Inverse(double *M, double *P, double res)
{
    int tid, tx, ty;
    tx = blockDim.x * blockIdx.x + threadIdx.x;
    ty = blockDim.y * blockIdx.y + threadIdx.y;

    int DimX = gridDim.x * blockDim.x;
    tid = DimX * ty + tx;
    
    P[tid] = M[tid]/res;
}