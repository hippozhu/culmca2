#include "mycommon.h"
#include "SVMData.h"
#include "Neighborhood.h"
#include "KernelNeighborhood.h"


KernelNeighborhood::KernelNeighborhood(SVMData& train, SVMData& test, int nc, int kk, int k1):Neighborhood(train, test, nc, kk, k1), gamma(1.0/nfeat){
  generateKernelMatrix();
  generateTestKernelMatrix();
  initE();
  initTriplets();
  deviceInitKernelMatrix(&train.ninst, &test.ninst, &train.nfeat, train.data_col, test.data_col);
}

void KernelNeighborhood::generateKernelMatrix(){
  K.resize(ninst, ninst);
  VectorXd vi(nfeat);
  VectorXd vj(nfeat);
  for(int i = 0; i < ninst; i++){
	(*sd).getVector(i, vi);
    for(int j = 0; j < ninst; j++){
	  if(i==j)
	    K(i, j) = 1.0;
	  else if(i>j)
	    K(i, j) = K(j, i);
	  else{
	    (*sd).getVector(j, vj);
	    K(i, j) = kernel_rbf(vi, vj);
	  }
	}
  }
}

void KernelNeighborhood::generateTestKernelMatrix(){
  K_test.resize(sd->ninst, sd_test->ninst);
  VectorXd vi(nfeat);
  VectorXd vj(nfeat);
  for(int i = 0; i < sd->ninst; i++){
	sd->getVector(i, vi);
    for(int j = 0; j < sd_test->ninst; j++){
	  sd_test->getVector(j, vj);
	  K_test(i, j) = kernel_rbf(vi, vj);
	}
  }
}

double KernelNeighborhood::kernel_rbf(VectorXd& vi, VectorXd& vj){
  return exp(-gamma * (vi - vj).squaredNorm());
}

void KernelNeighborhood::initE(){
  for(int i = 0; i < ninst*ninst; i ++){
    VectorXd v;
	E.push_back(v);
  }
}

VectorXd& KernelNeighborhood::getE(int i, int j){
  if (E[i * ninst + j].rows() == 0){
    E[i * ninst + j] = K.col(i) - K.col(j);
	E[j * ninst + i] = - E[i * ninst + j];
  }
  return E[i * ninst + j];
}

double KernelNeighborhood::dist(int i, int j, MatrixXd& omege){
  return (omege * getE(i, j)).squaredNorm();
}

double KernelNeighborhood::distance(int i, int j, MatrixXd& omege){
  return (omege * (K_test.col(i) - K.col(j))).squaredNorm();
}

void KernelNeighborhood::initTriplets(){
  for(int i = 0; i < ninst; i++)
	for(int j = 0; j < nn[sd->inst[i].label]; j++)
	  for(int l = 0; l < ninst; l++)
	    if(inOpposingClass(i, l)){
		  VTriplet vt = {i, getTargetByOffset(i, j), l, .0, false};
		  triplets.push_back(vt);
		}
  cout << "triplets #: " << triplets.size() << endl;
}