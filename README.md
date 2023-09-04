# GPU을 이용한 병렬처리 역행렬 연산 및 Determinant 계산


-	선형대수학에서 널리 사용되는 행렬의 계산 중 역행렬 계산은 계산 과정이 복잡하고 다양함.

-	3X3, 4X4 등 행렬의 크기가 작은 경우 사람이 수기로 계산할 수 있음.

-	하지만 행렬의 크기가 커져 10X10, 100X100 행렬이 될 경우 사람이 수기로 계산하기에 한계가 있음.

-	컴퓨터의 CPU를 이용하여 계산을 할 경우 연산이 가능하지만 일정 크기 이상의 행렬을 계산하게 될 경우 소요시간이 급격히 커짐.

-	따라서 GPU를 이용한 병렬처리를 사용하여 연산을 진행하고 소요시간을 줄이고자 함.



2. 진행과정

2-0.  CPU / GPU 역행렬 계산과정

Step1. 행렬의 행렬식(detA) 구하기
Step2. 수반행렬 구하기
Step3. 여인수행렬 구하기 ( 수반행렬의 전치 )
Step4. Step1에서 구한 행렬식 값을 여인수행렬의 원소에 나누어 줌

따라서 최종결과 역행렬 = (여인수행렬) * (1/detA)

2-1. InverseM_by_cpu.c

  
	
2-1-1. GetMatrix()를 통해 size*size 크기의 행렬을 만들고, 행렬원소를 랜덤값으로 세팅하여 M에 저장 
2-1-2. tmpM에 M 행렬 복사
2-1-3. 행렬 tmpM에 행렬식 구하기 by detM()

	
행렬식 구하는 방법에는 행렬을 상삼각화행렬하여 대각성분의 곱을 계산하여 구하는 방법과, (size-1)*(size-1)행렬을 만들어 행렬식을 구하는 방법 두가지가 있다.

후자의 방법은 (size-1)*(size-1)행렬을 2*2 matrix가 될 때까지 행렬내부로 들어가 2*2 matrix가 만들어지면 그 값을 3*3 matrix에 리턴하고, 4*4, 5*5 … size*size까지 리턴하는  방식으로 이루어지는데 이러한 방법으로 코드를 짜게 되면 재귀적으로 함수를 호출하고 계산하는 과정에서 상당히 오랜 시간이 걸린다.

따라서 우리는 전자의 방식을 사용하여 행렬식을 구하였다.

detM()함수에서 기본행연산을 통해 행렬을 상삼각화 해주었는데 만약 ratio 값의 분모가 0인경우 0을 0.001과 같이 작은값으로 세팅하여 분모가 0이되어 ratio 값이 nan, inf값이 되는 것을 방지해주었다.


 	 
	

2-1-4. 여인수행렬 구하기


CofactorMat() 함수를 통해 여인수행렬을 구해주었다. CofactorMat() 함수는 size*size만큼의 double형 배열을 할당해준 후 cofactor()함수를 통해 여인수값을 생성하여 인덱스에 리턴해주는 매커니즘으로 작동하는데 한 가지 고려해야 할 점이 있다. M(ij) (i=행번호, j=열번호) 에서 i+j 가 짝수이면 여인수값은 +값을 가지고, i+j가 홀수이면 여인수값은 – 값을 가진다.

따라서 Cofactor()를 통해 여인수값을 계산하고 경우 (짝수, 홀수) 를 나누어 리턴값을 설정해주었다.

Cofactor()의 여인수값의 계산은 MinortMat(M,size,I,j)를 호출하여 소행렬에 대한 행렬식값을 앞서 설명한 detM을 이용하여 구한 값이다.

 


2-1-5. MinorMat(M,size,i,j)는 i행 j열을 제외한 소행렬을 리턴하여 주는 함수이다. 다음과 같이 넘겨받은 i, j값에 대해 for문을 이용하여 해당 행,열을 제외한 행렬을 리턴하는 방식으로 코드를 구성하였다.


 

2-1-6. 앞서 구한 여인수행렬을 전치하여 리턴해 주는 함수 TransposeMat()를 이용하여 전치 여인수행렬에 대한 전치행렬인 수반행렬을 구해주었다.

2-1-7. 수반행렬의 원소에 2-1-3에서 구한 행렬식의 값을 나누어 주어 역행렬을 계산하였다.




	
2-2. InverseM_by_gpu.cu
	
 

2-2-1. 행렬 기본 설정

1) 역행렬 연산에 필요한 행렬 크기를 Matrixsize로 설정
2)  cudaMalloc을 이용하여 device에 사용할 메모리를 배열형태로 할당
3  할당된 배열 형태의 메모리에 rand() 함수 이용하여 행렬 원소 설정

 
2-2-2. tmpM에 행렬식 구하기 by getDeterminant()

행렬식 구하기에 앞서 cudaMemcpy() 를 이용하여 앞으로 device에서 쓰일 dev_origin, dev_M에 tmpM값을 복사해줬다.
행렬식을 구하는 방법에는 2가지가 있다. 앞서 CPU에서 사용한 상삼각행렬화를 통해 대각성분의 곱을 계산하는 방법을 사용하여 진행할 것이다.
먼저 getDeterminant를 호출하여 1블록당 1개의 스레드를 할당해주고 인자로 dev_M과 size를 넣어준다.
그 후 getDeterminat 함수 내에서 반복문을 통해 행렬의 사이즈만큼 sumRows 함수를 호출하게 해준다.
호출한 sumRows 함수는 한블록당 size개의 스레드를 할당해주고 함수의 작동 내용은 기본행연산을 통한 상삼각행렬화이다.
병렬처리를 이용한 기본행연산의 내용은 CPU에서 n! 횟수의 계산을 스레드를 이용하여 한번에 처리하는 방법이다.
여기서 CPU 구현 내용의 ratio 변수와 같은 일을 하는 factor 변수를 선언해줘 분모가 0이 되어 값이 nan,inf 가 되는 걸 방지하기 위해 연산결과에 지장을 주지않는 작은 값인 0.0001로 설정해주었다.
getDeterminant의 연산결과가 dev_M에 저장되고 그 값을 tmpM에 복사해준다음 tmpM의 대각성분을 모두 곱해주어 그 결과를 res에 저장한다. 구한 res 값이 행렬식 값이다.
 
 

2-2-3. 여인수행렬 구하기

먼저 CofactorMat()를 통해 여인수전개 후 연산 결과 값을 넣어줄 행렬을 생성해준다.
그다음 cofactor 함수를 이용하여 여인수전개를 진행한다. 여인수전개를 하기위해 필요한 소행렬식을 구하기 위해서
cofactor 함수 내에서 MinorMat 함수를 호출한다.
MinorMat 함수에서는 기존 행렬에서 한 행과 한 열을 선택해 제외하고 나머지 부분을 행렬로 생성하는데 이로인해 생성되는 행렬이 소행렬이다.
MinorMat 함수를 수행하여 얻은 Minor값(소행렬)을 이용하여 getDeterminant 함수, sumRows 함수를 수행한다.
위 두 함수의 수행내용은 앞서 설명한 행렬식 구하기의 과정과 동일하다.
행렬식 구하기의 과정을 수행한 뒤 상삼각화행렬화된 소행렬의 대각성분들의 곱을 res에 저장한다.
여기서 원행렬의 row,col 값의 합이 짝수 이면 res 를 리턴하고, 홀수일 경우 - res를 리턴해준다.
리턴해준 res 값을 행렬 N의 원소값에 대입하여 여인수행렬 N을 생성하고 결과적으로 dev_X에 저장된다.


 
2-2-4 수반행렬 / 역행렬 구하기

앞서 구한 여인수행렬 dev_X를 이용하여 행렬 원소를 전치해주는 Trans함수를 호출해 dev_Y에  전치된 행렬을 저장해주었다.
dev_Y = 수반행렬
구한 dev_Y(수반행렬)과 앞서 구한 행렬식을 이용하여 수반행렬의 원소들에 행렬식을 곱해주는 Inverse함수를 호출하여 역행렬을 구해줬다.




3. 프로젝트 결과

1)	Matrix Size = 3		

CPU	GPU
 	 

Matrix Size 가 다음과 같이 엄청 작은 경우 CPU 연산속도가 GPU 속도보다 빠르다.



2)	Matrix Size = 10


CPU	GPU
 	 
마찬가지로 Matrix Size가 10인 경우에도 GPU의 Global memory에 대한 접근속도 때문에 CPU의 연산속도가 더 빠르다.



3)	70 <= Matrix size <= 100


Matrix Size가 커질수록 연산 속도에서 확연한 성능차이가 발생하는 것을 확인할 수 있다.

