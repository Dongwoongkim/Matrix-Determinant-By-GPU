#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

double cofactor(double *M, int size, int i, int j);   // i행 j열에 대한 cofactor 값
double detM(double *M, int size); // 행렬식 by 상삼각행렬
double* MinorMat(double *M, int size, int row, int col); // i행 j열에 대한 소행렬 구하기
void PrintMatrix(double *M, int size12); // 행렬 표준 출력
double* GetMatrix(int size);   // 표준입력으로 행렬 얻기 
double* CofactorMat(double *M, int size); // 수반행렬 구하기
double* TransposeMat(double *M, int size); // 전치 
double* copyMatrix(double *M, int size); // 행렬복사 

int main()
{
    int size,sizeM;
    int Matrixsize;

    double ratio, dM;

    for(int size=2;size<=3;size++)
    {
        Matrixsize = size*size;
        int Buffersize = Matrixsize * sizeof(double);
        
        double *M = GetMatrix(size);
        double *tmpM = (double*)malloc(Buffersize);

        tmpM = copyMatrix(M,size);

        PrintMatrix(tmpM,size);
        printf("\n");
        
        clock_t start = clock();
        dM = detM(tmpM,size);
        
        printf("Matrix Size : %d / det of M : %f \n",size,dM);

        if(dM!=0)
        {
            // step2. 여인수행렬 구하기 
            double *cfMat = (double*) malloc(sizeof(double)*size*size);
            cfMat = CofactorMat(M,size);
            // printf("\n");

            // step3. 여인수행렬T -> 수반행렬 구하기
            double *TrM = (double*) malloc(sizeof(double)*size*size);
            TrM = TransposeMat(cfMat,size);

            free(cfMat);

            // printf("M의 역행렬 \n");
            // step4.  / det = 역행렬 
            double *N = (double*) malloc(sizeof(double)*size*size);

            for(int i=0;i<size*size;i++)
            {
                N[i] = TrM[i] / dM;
            }
            free(TrM);
            PrintMatrix(N,size);
        
            clock_t end = clock();
            printf("Time : %lfs\n", (double)(end - start) / CLOCKS_PER_SEC);

        }
    }
    return 0;
}

double* copyMatrix(double *M, int size)
{
    double *N = (double*)malloc(size*size*sizeof(double));

    for(int i=0;i<size*size;i++)
    {
        N[i] = M[i];
    }
    return N;
}

// 행렬 출력 함수
void PrintMatrix(double *M, int size)
{
    for(int i=0;i<size*size;i++)
    {
        printf(" %.2f ",M[i]);
        if( (i+1) % size ==0 ) printf("\n"); 
    }
    printf("\n");
}

double* MinorMat(double *M, int size, int row, int col)
{
    double *MinorM = malloc(sizeof(double)*(size-1)*(size-1));
    int cnt=0;
    
    for(int i=0;i<size;i++)
    {   
        if(i==row) continue;
        for(int j=0;j<size;j++)
        {
            if(j==col) continue;

            else
            {
                MinorM[cnt++] = M[i*size+j];
            }
        }
    }
    return MinorM;
}


double cofactor(double *M, int size, int i, int j)
{   
    // i는 행 , j는 row

    // M(ij)에서 i+j 가 짝수인 경우
    if((i+j)%2==0)
    {
        return detM(MinorMat(M,size,i,j),size-1);
    }
    
    // M(ij)에서 i+j가 홀수인 경우
    else
    {
        return -1.0 * detM(MinorMat(M,size,i,j),size-1);
    }
}

double* GetMatrix(int size)
{
    double *M = malloc(sizeof(double)*size*size);
    for(int i=0;i<size*size;i++)
    {
        M[i] = rand()%3;
    }
    return M;
}

double detM(double *M, int size)
{
    double ratio;
    double det=1;

    for(int i=0; i<size; i++)
    {   
        /*Upper triangular matrix using Gauss elimination method*/
        for(int j=i+1;j<size;j++)                /*j increases from i+1 to n*/
        {   
            if(M[i * size + i] == 0 ) M[i * size + i] = 0.001;
            ratio = M[j*size+i]/M[i*size+i];    /*Ratio is of element */

            for(int k=0;k<size;k++)              /*k starting from 0 to n*/
            {
            M[j*size+k]= M[j*size+k]- ratio * M[i*size+k];      /*Subtracting the product of ratio and element from the upper element*/
            }
        }
    }

    /*Calculating the determinant*/

    for(int i=0;i<size;i++)
    {
        det = det * M[i*size+i];   /*Multiplying the diagonal elements*/
    }

    return det;

}

double* CofactorMat(double *M, int size)
{
    double *CF = malloc(sizeof(double)*size*size);

    for(int i=0;i<size;i++)
    {
        for(int j=0;j<size;j++)
        {
            CF[i*size+j] = cofactor(M,size,i,j);
        }
    }
    return CF;
}

double* TransposeMat(double *M, int size)
{   
    double* TransM = malloc(sizeof(double)*size*size);

    for(int i=0;i<size;i++)
    {
        for(int j=0;j<size;j++)
        {
            TransM[i*size+j] = M[j*size+i];
        }
    }

    return TransM;
}
