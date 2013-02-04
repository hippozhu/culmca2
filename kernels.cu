#include "mycommon.h"

#define BSIZE 256

__constant__ int nfeat;
__constant__ int ntrain;
__constant__ int ntest;
__constant__ int nclass;
__constant__ int k;
__constant__ int nnegibor;
__constant__ double mu;
__constant__ double nu;
__constant__ int idx_o;

__constant__ int *target;
__constant__ double *km_train;
__constant__ double *km_test;
__constant__ double *O[2];
__constant__ double *t_target;
__constant__ double *t_triplet;
__constant__ double *t_update;
__constant__ double *t_gradient;

__constant__ short *label_train;
__constant__ short *label_test;
__constant__ struct Inst *grouped_inst;
__constant__ unsigned typecount[4];
__constant__ int *target_offset;
__constant__ int nn[4];

__constant__ double *dist_target;
__constant__ double *dist1;
__constant__ double *dist2;
__constant__ double *hinge_val;

__constant__ double *dist_knn;
__constant__ int *ino_knn;
__constant__ int *neighbor_knn;

__device__ double f_val;
__device__ double sub_fval[84];
__device__ double acc_knn;
__device__ int hits[4];

__device__ void kernelMatrix(double *km, double *d1, int n1, double *d2, int n2){
  int ub = n1 * n2;
  int stride = blockDim.x * gridDim.x;
  double c_val;
  int i, j;
  for (int m = blockIdx.x * blockDim.x + threadIdx.x; m < ub; m += stride){
    i = m / n2;
	j = m % n2;
	c_val = .0;
	for (int n = 0; n < nfeat; ++ n)
	  c_val += pow(d1[n * n1 + i] - d2[n * n2 + j], 2);
	km[m] = exp(-c_val / nfeat);
  }
}

__global__ void calcKM(double *train, double *test){
  kernelMatrix(km_train, train, ntrain, train, ntrain);
  kernelMatrix(km_test, test, ntest, train, ntrain);
}

__device__ double getElement(double *m, int i, int j, int stride){
  return *(m + i * stride + j);
}

__device__ void setElement(double *m, int i, int j, int stride, double val){
  m[i * stride + j] = val;
}

__device__ int getElementInt(int *m, int i, int j, int stride){
  return *(m + i * stride + j);
}

__device__ void setElementInt(int *m, int i, int j, int stride, int val){
  m[i * stride + j] = val;
}

__device__ int getTarget(int i, int kk){
  return target[i * k + kk];
}

__device__ int getTargetByOffset(int ino, int kk){
  return target[target_offset[ino] + kk];
}

__device__ int getTargetDist(int ino, int kk){
  return dist_target[target_offset[ino] + kk];
}

__device__ double calcDist(int i, double *km1, int j, double *km2){
  int tid = threadIdx.x;
  
  __shared__ double diff_k[256];
  __shared__ double sum[256];
  __shared__ double norm[64];
  
  if (tid < 64)
    norm[tid] = .0;
  
  int pos;
  for (int m = 0; m < (ntrain - 1)/blockDim.x + 1; ++ m){  
	__syncthreads();	
    pos = m * blockDim.x + tid;
	if (pos < ntrain)
      diff_k[tid] = getElement(km1, i, pos, ntrain) - getElement(km2, j, pos, ntrain);
	for (int d = 0; d < nfeat; ++ d){
	  __syncthreads();
	  if (pos < ntrain)
	    sum[tid] = getElement(O[idx_o], d, pos, ntrain) * diff_k[tid];
	  else
	    sum[tid] = .0;
		
	  int stride = blockDim.x/2;
	  while (stride > 0){
	    __syncthreads();
		if (tid < stride)
		  sum[tid] += sum[tid + stride];
		stride /= 2;
	  }
	  __syncthreads();
	  
	  if (tid == 0)
	    norm[d] += sum[0];
	}
  }
  
  if (tid < nfeat)
    norm[tid] = norm[tid]*norm[tid];
  
  __syncthreads();
  
  double s = .0;
  for (int d = 0; d < nfeat; ++ d)
	s += norm[d];
  return s;
}

__device__ void calcTargetDist(){
  int tid = threadIdx.x;
  int bid = blockIdx.x; 
  int i, j;
  if (tid == 0)
    sub_fval[bid] = .0;

  int c = 0;
  for (int m = 0; m < ntrain; ++ m)
    for (int n = 0; n < nn[label_train[m]]; ++ n){
	  i = m;
	  j = getTargetByOffset(m, n);
	  if(c%gridDim.x == bid){
	  double val = calcDist(i, km_train, j, km_train);
      if (tid == 0){
	    dist_target[target_offset[m] + n] = val;
        sub_fval[bid] += val;
	  }
	  }
	  ++ c;
	}
}

__device__ void updateDist(double *dist, struct Inst * inst1, int height, struct Inst * inst2, int width){
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int i, j;
  for (int m = bid; m < height * width; m += gridDim.x){
    i = inst1[m / width].ino;
	j = inst2[m % width].ino;
	double val = calcDist(i, km_train, j, km_train);
    if (tid == 0)
	  dist[m] = val;
  }
}

__global__ void update2(){
  calcTargetDist();
  updateDist(dist1, grouped_inst, typecount[0], grouped_inst + typecount[0], typecount[1]);
}

__global__ void zeroHinge(){
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int size = gridDim.x * blockDim.x;
  for (int m = blockDim.x * bid + tid; m < typecount[0] * typecount[1] * 2 * k; m += size)
    hinge_val[m] = .0; 
}

__device__ double hinge(double s){
  if (s <= -1.0)
    return .0;
  else if (s >= 0)
    return 1.0;
  else
    return 1 + s;
}

__device__ void updateTri(int idx1, int idx2, int idx3, double h){
  __syncthreads();
  for (int p = threadIdx.x; p < ntrain; p += blockDim.x)
    t_triplet[p * ntrain + idx1] += h * (getElement(km_train, idx2, p, ntrain) - getElement(km_train, idx3, p, ntrain));
}

__global__ void zeroT_triplet(){
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int size = gridDim.x * blockDim.x;
  for (int m = blockDim.x * bid + tid; m < ntrain * ntrain; m += size)
    t_triplet[m] = .0;  
}

__global__ void update3_2(){
  int bid = blockIdx.x;
  int i, j, l;
  double vdist, h;
  
  if (bid == 0 && threadIdx.x == 0)
    f_val = .0;
  
  for (int m = 0; m < typecount[TN] * typecount[TP]; ++ m){
    for (int kk = 0; kk < nn[grouped_inst[m / typecount[TP]].label]; ++ kk){
      i = grouped_inst[m / typecount[TP]].ino;
	  l = grouped_inst[typecount[TN] + m % typecount[TP]].ino;
	  //j = getTarget(i, kk);
	  j = getTargetByOffset(i, kk);
	  //vdist = 1 + getElement(dist_target, i, kk, k) - dist1[m];
	  //double tmp = getTargetDist(i, kk);
	  //vdist = 1 + getTargetDist(i, kk) - dist1[m];
	  vdist = 1 + dist_target[target_offset[i] + kk] - dist1[m];
	  if (vdist > 0 && blockIdx.x == 0 && threadIdx.x == 0)
	    f_val += vdist;
      h = hinge(vdist);
	  if (h > 0){
	    if (label_train[i] == TP)
		  h *= nu;
	    if (i % gridDim.x == bid)
		  updateTri(i, l, j, h);
	    if (j % gridDim.x == bid)
		  updateTri(j, j, i, h);
	    if (l % gridDim.x == bid)
		  updateTri(l, i, l, h);
	  }
	  
	  
      l = grouped_inst[m / typecount[TP]].ino;
	  i = grouped_inst[typecount[TN] + m % typecount[TP]].ino;
	  //j = getTarget(i, kk);
	  j = getTargetByOffset(i, kk);
	  //vdist = 1 + getElement(dist_target, i, kk, k) - dist1[m];
	  vdist = 1 + dist_target[target_offset[i] + kk] - dist1[m];
	  if (vdist > 0 && blockIdx.x == 0 && threadIdx.x == 0)
	    f_val += vdist;
      h = hinge(vdist);
	  if (h > 0){
	    if (label_train[i] == TP)
		  h *= nu;
	    if (i % gridDim.x == bid)
		  updateTri(i, l, j, h);
	    if (j % gridDim.x == bid)
		  updateTri(j, j, i, h);
	    if (l % gridDim.x == bid)
		  updateTri(l, i, l, h);
	  }
	}
  }
}

__global__ void calcFval(){ 
  if (blockIdx.x == 0 && threadIdx.x == 0)
	for (int i = 0; i < gridDim.x; ++ i)
	  f_val += sub_fval[i];
}

__global__ void updateUpdateTerm(double alpha){
  int size = gridDim.x * blockDim.x;
  for (int m = blockIdx.x * blockDim.x + threadIdx.x; m < ntrain * ntrain; m += size){
    if (m/ntrain == m%ntrain)
      t_update[m] = 1 - 2 * alpha * (t_target[m] + mu * t_triplet[m]);
      //t_update[m] = 1 - 2 * alpha * ((1-mu) * t_target[m] + mu * t_triplet[m]);
	else
      t_update[m] = - 2 * alpha * (t_target[m] + mu * t_triplet[m]);
      //t_update[m] = - 2 * alpha * ((1-mu) * t_target[m] + mu * t_triplet[m]);
  }
}

__global__ void zeroO(){
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int size = gridDim.x * blockDim.x;
  for (int m = blockDim.x * bid + tid; m < nfeat * ntrain; m += size)
    O[1 - idx_o][m] = .0; 
}

__global__ void updateO(){
  int tid = threadIdx.x;
  int bid_row = blockIdx.x;
  int bid_col = blockIdx.y;
  
  __shared__ double o_element[BSIZE];
  
	if (bid_col * BSIZE + tid < ntrain){
  for (int start = 0; start < ntrain; start += BSIZE){
	int len = min(BSIZE, ntrain - start);
    if(tid < len)
      o_element[tid] = getElement(O[idx_o], bid_row, start + tid, ntrain);
	__syncthreads();
	
    for (int i = 0; i < len; ++ i){
	  double val = o_element[i] * getElement(t_update, i + start, bid_col * BSIZE + tid, ntrain);
	  //__syncthreads();
	  O[1 - idx_o][bid_row * ntrain + bid_col * BSIZE + tid] += val;
	}
  }
	}
}

__global__ void updateO1(){
  int tid = threadIdx.x;
  int bid_row = blockIdx.x;
  int bid_col = blockIdx.y;
  int workingtid = min(BSIZE, ntrain - bid_col * BSIZE);
  
  if (tid < workingtid)
    O[1 - idx_o][bid_row * ntrain + bid_col * BSIZE + tid] = .0;
  
  //__shared__ double o_element[BSIZE];
  
  for (int start = 0; start < ntrain; start += BSIZE){
	int len = min(BSIZE, ntrain - start);
    //if(tid < len)
    //  o_element[tid] = getElement(O[idx_o], bid_row, start + tid, ntrain);
	//__syncthreads();
	
    for (int i = 0; i < len; ++ i){
	if (tid < workingtid){
	  double val = getElement(O[idx_o], bid_row, start + i, ntrain) * getElement(t_update, i + start, bid_col * BSIZE + tid, ntrain);
	  //double val = o_element[i] * getElement(t_update, i + start, bid_col * BSIZE + tid, ntrain);
	  //__syncthreads();
	  O[1 - idx_o][bid_row * ntrain + bid_col * BSIZE + tid] += val;
	}
	}
  }
}

__global__ void knnUpdateDist(){
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int size = gridDim.x;
	
  for(int m = bid; m < ntest * ntrain; m += size){
    int i = m / ntrain;
	int j = m % ntrain;
	double d = calcDist(i, km_test, j, km_train);
	if (tid == 0){
	  ino_knn[m] = j;
      dist_knn[m] = d;
	}
  }
}

// lauched with # block = ntest
__global__ void knnFindNeighbor(){
  
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int len = ntrain / BSIZE;
  int start = tid * len;
  if (tid < ntrain % BSIZE){
    start += tid;
	++ len;
  }
  else
    start += ntrain % BSIZE;
  
  __syncthreads();
  //int b = min(len, k);
  int b = min(len, nnegibor);
  for (int i = 0; i < b; ++ i)
    for (int j = start; j < start + len - i - 1; ++ j)
	  if(getElement(dist_knn, bid, j, ntrain) < getElement(dist_knn, bid, j + 1, ntrain)){
	    double tmp_dist = getElement(dist_knn, bid, j, ntrain);
		setElement(dist_knn, bid, j, ntrain, getElement(dist_knn, bid, j + 1, ntrain));
		setElement(dist_knn, bid, j + 1, ntrain, tmp_dist);
		
		int tmp_ino = getElementInt(ino_knn, bid, j, ntrain);
		setElementInt(ino_knn, bid, j, ntrain, getElementInt(ino_knn, bid, j + 1, ntrain));
		setElementInt(ino_knn, bid, j + 1, ntrain, tmp_ino);
	  }

  __syncthreads();  

  __shared__ double dist[BSIZE];
  __shared__ int ino[BSIZE];
  __shared__ int shortest[BSIZE];
  
  int p = start + len -1;
  //for (int i = 0; i < k; ++ i){
  for (int i = 0; i < nnegibor; ++ i){
    if (b > 0){
      dist[tid] = getElement(dist_knn, bid, p, ntrain);
      ino[tid] = getElementInt(ino_knn, bid, p, ntrain);
	}
	else
      dist[tid] = DBL_MAX;
	
    shortest[tid] = tid;
  
	int stride = blockDim.x/2;
	while (stride > 0){
	  __syncthreads();
	  if (tid < stride){
		if (dist[tid] > dist[tid + stride]){
		  dist[tid] = dist[tid + stride];
		  ino[tid] = ino[tid + stride];
		  shortest[tid] = shortest[tid + stride];
		}
	  }
	  stride /= 2;
	}
	
	__syncthreads();
	if(tid == 0)
	  //setElementInt(neighbor_knn, bid, i, k, ino[0]);
	  setElementInt(neighbor_knn, bid, i, nnegibor, ino[0]);
	if(tid == shortest[0]){
	  -- b;
	  -- p;
	}
  }
}

__global__ void knnMatching(){
  //int ub = ntest * k;
  int ub = ntest * nnegibor;
  int stride = blockDim.x * gridDim.x;  
  int idx_test, idx_train;
  for (int m = blockIdx.x * blockDim.x + threadIdx.x; m < ub; m += stride){
    //idx_test = m / k;
    idx_test = m / nnegibor;
	idx_train = neighbor_knn[m];
	if (label_test[idx_test] == label_train[idx_train])
	  neighbor_knn[m] = 1;
	else
	  neighbor_knn[m] = 0;
  }
}

// lauch with single block
__global__ void knnAcc(int neiborhood_size){
  int tid = threadIdx.x;
  int stride = blockDim.x;
  
  if (tid < 4)
    hits[tid] = 0;
	
  __shared__ int matched[BSIZE];
  matched[tid] = 0;
  
  for (int m = tid; m < ntest; m += stride){
    int nsametype = 0;
    for (int i = 0; i < neiborhood_size; ++ i)
	  //nsametype += neighbor_knn[m * k + i];
	  nsametype += neighbor_knn[m * nnegibor + i];
	if (nsametype > neiborhood_size/2){
	  matched[tid] += 1;
	  if (label_test[m] == TP)
	    atomicAdd(&hits[TP], 1);
	}
	else{
	  if (label_test[m] == TN)
	    atomicSub(&hits[TN], 1);
	}
  }
  
  int stride1 = blockDim.x/2;
  while (stride1 > 0){
	__syncthreads();
	if (tid < stride1)
	  matched[tid] += matched[tid + stride1];
	stride1 /= 2;
  }
  
  __syncthreads();  
  if (tid ==0)
    acc_knn = 1.0 * matched[0] / ntest;
}

void deviceInitKernelMatrix(int *trainninst, int *testninst, int *nf, double *traindata, double *testdata){

  cudaMemcpyToSymbol(ntrain, trainninst, sizeof(int), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(ntest, testninst, sizeof(int), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(nfeat, nf, sizeof(int), 0, cudaMemcpyHostToDevice);
  
  
  double *d_train_data, *d_test_data;
  cudaMalloc((void **)&d_train_data, sizeof(double) * (*trainninst) * (*nf));
  cudaMalloc((void **)&d_test_data, sizeof(double) * (*testninst) * (*nf));
  cudaMemcpy(d_train_data, traindata, sizeof(double) * (*trainninst) * (*nf), cudaMemcpyHostToDevice);
  cudaMemcpy(d_test_data, testdata, sizeof(double) * (*testninst) * (*nf), cudaMemcpyHostToDevice);
  
  double *d_kernel_matrix_train, *d_kernel_matrix_test;
  cudaMalloc((void **)&d_kernel_matrix_train, sizeof(double) * (*trainninst) * (*trainninst));
  cudaMemcpyToSymbol(km_train, &d_kernel_matrix_train, sizeof(double*), 0, cudaMemcpyHostToDevice);
  cudaMalloc((void **)&d_kernel_matrix_test, sizeof(double) * (*testninst) * (*trainninst));
  cudaMemcpyToSymbol(km_test, &d_kernel_matrix_test, sizeof(double*), 0, cudaMemcpyHostToDevice);
  
  // Run the event recording
  cudaEvent_t start_event, stop_event;
  cudaEventCreate(&start_event) ;
  cudaEventCreate(&stop_event) ;
  cudaEventRecord(start_event, 0);
  
  calcKM<<<84, 256>>>(d_train_data, d_test_data);
  cudaThreadSynchronize();
  
  cudaEventRecord(stop_event, 0);
  cudaEventSynchronize(stop_event);
  
  cudaFree(d_train_data);
  cudaFree(d_test_data);
}

void deviceInitTarget(int *h_target, int trainninst, int targetsize, int *kk, int *nc, int *Nneighbor, int *offset){
  int *d_target;
  cudaMalloc((void **)&d_target, sizeof(int) * targetsize);
  cudaMemcpy(d_target, h_target, sizeof(int) * targetsize, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(target, &d_target, sizeof(int*), 0, cudaMemcpyHostToDevice);

  
  cudaMalloc((void **)&d_target, sizeof(int) * trainninst);
  cudaMemcpy(d_target, offset, sizeof(int) * trainninst, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(target_offset, &d_target, sizeof(int*), 0, cudaMemcpyHostToDevice);
  
  cudaMemcpyToSymbol(k, kk, sizeof(int), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(nclass, nc, sizeof(int), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(nn, Nneighbor, sizeof(int) * 4, 0, cudaMemcpyHostToDevice);
}

void deviceInitLabelTrain(struct Inst *inst, unsigned ninst){
  short *label = new short[ninst];
  for (int i = 0; i < ninst; ++ i)
    label[i] = inst[i].label;
  
  short *d_label;
  cudaMalloc((void **)&d_label, sizeof(short) * ninst);
  cudaMemcpy(d_label, label, sizeof(short) * ninst, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(label_train, &d_label, sizeof(short*), 0, cudaMemcpyHostToDevice);
  delete[] label;
}

void deviceInitLabelTest(struct Inst *inst, unsigned ninst){
  short *label = new short[ninst];
  for (int i = 0; i < ninst; ++ i)
    label[i] = inst[i].label;
  
  short *d_label;
  cudaMalloc((void **)&d_label, sizeof(short) * ninst);
  cudaMemcpy(d_label, label, sizeof(short) * ninst, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(label_test, &d_label, sizeof(short*), 0, cudaMemcpyHostToDevice);
  delete[] label;
}

void deviceInitInstList(struct Inst *inst, unsigned *count, unsigned ninst, int nc, int kk, int targetsize){

  cudaMemcpyToSymbol(typecount, count, sizeof(unsigned) * 4, 0, cudaMemcpyHostToDevice);
  
  struct Inst *gi[4];
  for (int i = 0; i < 4; ++ i){
    if (count[i] > 0)
      gi[i] = (struct Inst *)malloc(sizeof(struct Inst) * count[i]);
  }

  //int *index = new int[ninst];
  int p[4] = {0, 0, 0, 0};
  for(int i = 0; i < ninst; ++ i){
    int type = inst[i].label;
	gi[type][p[type]].ino = inst[i].ino;
	gi[type][p[type]].label = inst[i].label;
	//index[i] = p[type];
	//for(int j = 0; j < inst[i].label; ++ j)
	  //index[i] += count[j];
	++ p[type];
  }
  
  struct Inst *d_inst;
  cudaMalloc((void **)&d_inst, sizeof(struct Inst) * ninst);
  unsigned start = 0;
  for (int i = 0; i < 4; ++ i){
    if (count[i] > 0)
	  cudaMemcpy(d_inst + start, gi[i], sizeof(struct Inst) * count[i], cudaMemcpyHostToDevice);
    start += count[i];
  }
  cudaMemcpyToSymbol(grouped_inst, &d_inst, sizeof(struct Inst *), 0, cudaMemcpyHostToDevice);

  for (int i = 0; i < 4; ++ i){
    if (count[i] > 0)
      free(gi[i]);
  }
  
  double *distanceTarget, *distanceMatrix1, *distanceMatrix2, *hinge_array;
  
  cudaMalloc((void **)&distanceTarget, sizeof(double) * targetsize);
  cudaMemcpyToSymbol(dist_target, &distanceTarget, sizeof(double *), 0, cudaMemcpyHostToDevice);
  
  if (nc == 2){
    cudaMalloc((void **)&distanceMatrix1, sizeof(double) * count[0] * count[1]);
	cudaMemcpyToSymbol(dist1, &distanceMatrix1, sizeof(double *), 0, cudaMemcpyHostToDevice);
	
    cudaMalloc((void **)&hinge_array, sizeof(double) * count[0] * count[1] * 2 * kk);
	cudaMemcpyToSymbol(hinge_val, &hinge_array, sizeof(double *), 0, cudaMemcpyHostToDevice);
  }
  else{
    cudaMalloc((void **)&distanceMatrix1, sizeof(double) * count[0] * count[3]);
    cudaMalloc((void **)&distanceMatrix2, sizeof(double) * count[1] * count[2]);
	cudaMemcpyToSymbol(dist1, &distanceMatrix1, sizeof(double *), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(dist2, &distanceMatrix2, sizeof(double *), 0, cudaMemcpyHostToDevice);
  }
  
}

void deviceInitMu(double m, double n){
  double local_m = m;
  cudaMemcpyToSymbol(mu, &local_m, sizeof(double), 0, cudaMemcpyHostToDevice);
  double local_n = n;
  cudaMemcpyToSymbol(nu, &local_n, sizeof(double), 0, cudaMemcpyHostToDevice);
}

void deviceInitO(double *o, int size){
  double *d_t;
  //cout << "double O: " << o[1] << endl;
  cudaMalloc((void **)&d_t, sizeof(double) * size);
  cudaMemcpy(d_t, o, sizeof(double) * size, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(O, &d_t, sizeof(double*), 0, cudaMemcpyHostToDevice);
  //cout << "d_t: " << d_t << endl;
  
  cudaMalloc((void **)&d_t, sizeof(double) * size);
  cudaMemcpyToSymbol(O, &d_t, sizeof(double*), sizeof(double*), cudaMemcpyHostToDevice);
  //cout << "d_t: " << d_t << endl;
}

void deviceInitTargetTerm(double *t, int size){
  double *d_t;
  cudaMalloc((void **)&d_t, sizeof(double) * size);
  cudaMemcpy(d_t, t, sizeof(double) * size, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(t_target, &d_t, sizeof(double*), 0, cudaMemcpyHostToDevice);
}

void deviceInitUpdateTerm(int size1, int size2){
  double *d_t;
  cudaMalloc((void **)&d_t, sizeof(double) * size1);
  cudaMemcpyToSymbol(t_update, &d_t, sizeof(double*), 0, cudaMemcpyHostToDevice);
  
  cudaMalloc((void **)&d_t, sizeof(double) * size2);
  cudaMemcpyToSymbol(t_gradient, &d_t, sizeof(double*), 0, cudaMemcpyHostToDevice);
}

void deviceInitTri(int size){
  double *t_o;
  cudaMalloc((void **)&t_o, sizeof(double) * size);
  cudaMemcpyToSymbol(t_triplet, &t_o, sizeof(double*), 0, cudaMemcpyHostToDevice);
}

void deviceInitKnn(int n_train, int n_test, int kk){
  double *d_knn;
  cudaMalloc((void **)&d_knn, sizeof(double) * n_test * n_train);
  cudaMemcpyToSymbol(dist_knn, &d_knn, sizeof(double*), 0, cudaMemcpyHostToDevice);
  
  int* i_knn;
  cudaMalloc((void **)&i_knn, sizeof(int) * n_test * n_train);
  cudaMemcpyToSymbol(ino_knn, &i_knn, sizeof(int*), 0, cudaMemcpyHostToDevice);  
  
  cudaMalloc((void **)&i_knn, sizeof(int) * n_test * kk);
  cudaMemcpyToSymbol(neighbor_knn, &i_knn, sizeof(int*), 0, cudaMemcpyHostToDevice);
  
  cudaMemcpyToSymbol(nnegibor, &kk, sizeof(int), 0, cudaMemcpyHostToDevice);
}

void kernelTest(int d, int n, int n_test, int kk, double *result, double mu, double alpha, double nu, int k1){
  double dd[20];
  int h_hits[4];
  deviceInitKnn(n, n_test, 40);
  //double f = DBL_MAX;
  double f_old = DBL_MAX;
  double min_iter = 0;
  
  double global_max_acc = .0;
  unsigned global_max_iter = 0;
  
  //bool reduced = true;
  int idx = 1;
  //zeroHinge<<<84, 256>>>();
  //zeroT_triplet<<<84, 256>>>(); 
  unsigned iter = 0; 
  while(true){
  
  // Run the event recording
  cudaEvent_t start_event, stop_event;
  cudaEventCreate(&start_event);
  cudaEventCreate(&stop_event);
  cudaEventRecord(start_event, 0);
  
  cout << endl << "Iter = " << iter << ", K = "<< kk << ", K1 = "<< k1 << ", mu = " << mu << ", nu = " << nu << endl;  

  idx = 1 - idx;
  cudaMemcpyToSymbol(idx_o, &idx, sizeof(int), 0, cudaMemcpyHostToDevice);
  
  // update distances to targets(i,j) and between opposing points(i,l)
  update2<<<84, 256>>>();
  // update t_triplet by calculating vdist of every (i, j, l)
  zeroT_triplet<<<84, 256>>>();  
  update3_2<<<84, 256>>>(); 
  //update3_3<<<84, 256>>>();  
  
  // update object function value
  calcFval<<<84, 256>>>();
  
  cudaThreadSynchronize();  
  cudaMemcpyFromSymbol(&dd[9], f_val, sizeof(double), 0, cudaMemcpyDeviceToHost);
  
  /*
  if (dd[9] < f)
    alpha *= 1.1;
  else
    alpha /= 2;
  f = dd[9];
  */
  cout << "f_val= " << dd[9];
  if (dd[9] < f_old){
	cout << ", reduced by " << f_old - dd[9] << endl;  
    f_old = dd[9];
	min_iter = iter;
	//reduced = true;
    alpha *= 1.1;
	
  knnUpdateDist<<<84, BSIZE>>>();
  knnFindNeighbor<<<n_test, BSIZE>>>();
  knnMatching<<<84, BSIZE>>>();  
  
  for (int i = 0; i < 20; ++ i){
    knnAcc<<<1, BSIZE>>>(2 * i + 1);
    cudaThreadSynchronize();
    cudaMemcpyFromSymbol(h_hits, hits, sizeof(int) * 4, 0, cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(&dd[i], acc_knn, sizeof(double), 0, cudaMemcpyDeviceToHost);
    //cout << dd[i] << "(" << h_hits[0] << "," << h_hits[1] << "), ";
    cout << h_hits[0] + h_hits[1] << "(" << h_hits[0] << "," << h_hits[1] << "), ";
  }
  
  double max_acc = .0;
  int max_acc_k = -1;
  for (int i = 0; i < 20; ++ i){
    if (dd[i] > max_acc){
	  max_acc = dd[i];
	  max_acc_k = 2 * i + 1;
	}
  }
  if (max_acc >= global_max_acc&&iter>10){
    global_max_acc = max_acc;
	global_max_iter = iter;
  }
  cout << endl << "max acc = " << max_acc << " at k = " << max_acc_k 
  << ". global max = " << global_max_acc << " at iter = " << global_max_iter;
  }
  else{
	cout << ", increased by " << dd[9] - f_old;
	//reduced = false;
    alpha /= 10;
  //int idx = iter % 2;
  //if (reduced)
    idx = 1 - idx;
    cudaMemcpyToSymbol(idx_o, &idx, sizeof(int), 0, cudaMemcpyHostToDevice);
    update2<<<84, 256>>>();
  
  // update t_triplet by calculating vdist of every (i, j, l)
  zeroT_triplet<<<84, 256>>>();  
  update3_2<<<84, 256>>>(); 
  //update3_3<<<84, 256>>>();  
  
  }
  
  cout << endl << "min_f = " << f_old << " at iter " << min_iter << ", alpha = " << alpha << endl;

  // t_update = I - 2 * alpha * (t_target + t_triplet)
  updateUpdateTerm<<<84, 256>>>(alpha);  
  
  // update omega = omega * t_update
  zeroO<<<84, 256>>>();
  dim3 dimGrid(d, (n - 1) / BSIZE + 1);
  dim3 dimBlock(BSIZE);
  updateO1<<<dimGrid, dimBlock>>>();  
  cudaThreadSynchronize();
  
  float time_kernel;
  cudaEventRecord(stop_event, 0);
  cudaEventElapsedTime(&time_kernel, start_event, stop_event);
  cout << "time " << time_kernel/1000 << endl;
  ++ iter;
  //if (iter > 100)
  if (alpha < 1e-10)
    break;
  }
}
  /*
  cudaEvent_t start_event1, stop_event1;
  cudaEventCreate(&start_event1);
  cudaEventCreate(&stop_event1);
  cudaEventRecord(start_event1, 0);
  cudaThreadSynchronize();
  
  float time_kernel1;
  cudaEventRecord(stop_event1, 0);
  cudaEventElapsedTime(&time_kernel1, start_event1, stop_event1);
  cout << "time1 " << time_kernel1/1000 << endl;
  */
