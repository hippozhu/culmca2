#include "mycommon.h"
#include "SVMData.h"
#include "Neighborhood.h"

Neighborhood::Neighborhood(SVMData& train, SVMData& test, int nc, int kk, int k1):sd(&train), sd_test(&test), nclass(nc), k(kk), nfeat(train.nfeat), ninst(train.ninst), ninst_test(test.ninst)
{
  nn[0] = k;
  nn[1] = k1;
  nn[2] = 0;
  nn[3] = 0;
  findTarget();
  deviceInitTarget(target, train.ninst, target_size, &k, &nclass, nn, target_offset);
  deviceInitLabelTrain(train.inst, train.ninst);
  deviceInitLabelTest(test.inst, test.ninst);
  deviceInitInstList(train.inst, train.typecount, train.ninst, nclass, k, target_size);
}

Neighborhood::~Neighborhood(){
  free(target);
}

double Neighborhood::dist(int i, int j, MatrixXd& m){
  return .0;
}

double Neighborhood::distance(int i, int j, MatrixXd& omege){
  return .0;
}

double Neighborhood::edist(int i, int j){
  double *p1 = (*sd).getDataPoint(i);
  double *p2 = (*sd).getDataPoint(j);
  double dist = 0.0;
  for (int l = 0; l < nfeat; l++)
    dist += pow(p1[l]-p2[l], 2);
  return dist;
}

void Neighborhood::calcEdistMatrix(double* distMatrix){
  for(int i = 0; i < ninst; i++)
    for(int j = 0; j < ninst; j++){
	  if(i==j)
	    distMatrix[i*ninst+j] = .0;
	  else if(i>j)
	    distMatrix[i*ninst+j] = distMatrix[j*ninst+i];
	  else{
	    distMatrix[i*ninst+j] = edist(i, j);
	  }
	}
}

void Neighborhood::calcDistMatrix(double* distMatrix, MatrixXd& M){
  for(int i = 0; i < ninst; i++)
    for(int j = 0; j < ninst; j++){
	  if(i==j)
	    distMatrix[i*ninst+j] = .0;
	  else if(i>j)
	    distMatrix[i*ninst+j] = distMatrix[j*ninst+i];
	  else{
	    distMatrix[i*ninst+j] = dist(i, j, M);
	  }
	}
}

void Neighborhood::calcDistMatrix(MatrixXd& distMatrix, MatrixXd& M){
  for(int i = 0; i < distMatrix.rows(); i++)
    for(int j = 0; j < distMatrix.cols(); j++){
	/*
	  if(i==j)
	    distMatrix(i, j) = .0;
	  else if(i>j)
	    distMatrix(i, j) = distMatrix(j, i);
	  else
	  */
	    distMatrix(i, j) = distance(i, j, M);
	}
}


void Neighborhood::findTarget(){
  target_offset = (int*)malloc(sizeof(int)*ninst);
  int typecount[4];
  for(int i = 0; i < 4; ++ i)
    typecount[i] = 0;
  for(int i = 0; i < ninst; ++ i)
    ++ typecount[sd->inst[i].label];
  target_size = 0;
  for(int i = 0; i < 4; ++ i)
    target_size += typecount[i] * nn[i];
  target = (int*)malloc(sizeof(int)*target_size);
  
  double *edistMatrix = (double*)malloc(sizeof(double)*ninst*ninst);
  calcEdistMatrix(edistMatrix);
  acd = .0;
  int base = 0;
  for(int i = 0; i < ninst; ++ i){
  
    target_offset[i] = base;
	base += nn[sd->inst[i].label];
	
    vector<DistPair> dp;
	for(int j = 0; j < ninst; ++ j){
	  if(i==j)
	    continue;
	  if(inSameClass(i,j)){
	    DistPair d = {j, edistMatrix[i * ninst + j]};
	    dp.push_back(d);
	  }
	}
	
	sort(dp.begin(), dp.end());
	
	for(int j = 0; j < nn[sd->inst[i].label]; ++ j)
	  target[target_offset[i] + j] = dp[j].ino;
	  
	acd += dp[0].dist;
  }
  acd /= ninst;
}

int Neighborhood::getTarget(int i, int t){
  return target[i * k + t];
}

int Neighborhood::getTargetByOffset(int ino, int kk){
  return target[target_offset[ino] + kk];
}

bool Neighborhood::inSameClass(int i, int j){
  return (*sd).inst[i].label == (*sd).inst[j].label;
}

bool Neighborhood::inOpposingClass(int i, int j){
  if (nclass==2)
    return (*sd).inst[i].label!=(*sd).inst[j].label;
  
  switch ((*sd).inst[i].label){
    case TN:
	  if ((*sd).inst[j].label == FN)
	    return true;
	  break;
    case TP:
	  if ((*sd).inst[j].label == FP)
	    return true;
	  break;
    case FP:
	  if ((*sd).inst[j].label == TP)
	    return true;
	  break;
    case FN:
	  if ((*sd).inst[j].label == TN)
	    return true;
	  break;
  }
  return false;
}

bool Neighborhood::inSameClass(Inst& i1, Inst& i2){
  return i1.label == i2.label;
}

bool Neighborhood::inOpposingClass(Inst& i1, Inst& i2){
  if (nclass==2)
    return i1.label!=i2.label;
  
  switch (i1.label){
    case TN:
	  if (i2.label == FN)
	    return true;
	  break;
    case TP:
	  if (i2.label == FP)
	    return true;
	  break;
    case FP:
	  if (i2.label == TP)
	    return true;
	  break;
    case FN:
	  if (i2.label == TN)
	    return true;
	  break;
  }
  return false;
}

bool Neighborhood::isType(int i, int type){
  return (*sd).inst[i].label == type;
}

void Neighborhood::dataPointToVector(double *p, VectorXd& v){
  for(int i = 0; i < nfeat; i++)
    v(i) = p[i];
}

double Neighborhood::mdist(int i, int j, MatrixXd& M){
  VectorXd vi(nfeat);
  VectorXd vj(nfeat);
  dataPointToVector((*sd).getDataPoint(i), vi);
  dataPointToVector((*sd).getDataPoint(j), vj);
  
  return (vi - vj).transpose() * M * (vi - vj);
}

MatrixXd Neighborhood::outerProduct(int i, int j){
  VectorXd vi(nfeat);
  VectorXd vj(nfeat);
  dataPointToVector((*sd).getDataPoint(i), vi);
  dataPointToVector((*sd).getDataPoint(j), vj);
  
  return (vi - vj) * (vi - vj).transpose();
}

double Neighborhood::violatedDist(int i, int j, int l, MatrixXd& M){
  //return acd + mdist(i, j, M) - mdist(i, l, M);
  return 1 + dist(i, j, M) - dist(i, l, M);
  //return mdist(i, j, M) - mdist(i, l, M);
}

double Neighborhood::weight(double dist){
  return exp(-dist/0.1);
}

double Neighborhood::knn(MatrixXd& M, int kk, bool initial){
  double *distMatrix = (double*)malloc(sizeof(double)*ninst*ninst);
  if(initial)
    calcEdistMatrix(distMatrix);
  else
    calcDistMatrix(distMatrix, M);
  double acc = .0;
  int count[4] = {0, 0, 0, 0};
  for(int i = 0; i < ninst; ++ i){
    vector<DistPair> dp;
	for(int j = 0; j < ninst; ++ j){
	  if(i==j || (!inSameClass(i, j) && !inOpposingClass(i, j)))
	    continue;
	  DistPair d = {j, distMatrix[i * ninst + j]};
	  dp.push_back(d);
	}
	
	sort(dp.begin(), dp.end());
	
	int similar_target_neighbor = 0;
	for(int j = 0; j < kk; ++ j){
	  if(inSameClass(i, dp[j].ino)){
	    ++ similar_target_neighbor;
	  }
	}
	
	if(similar_target_neighbor > kk/2){
	  acc += 1;
	  ++ count[(*sd).inst[i].label];
	}
  }
  //cout << "[" << count[0] << ", " << count[1] << ", " << count[2] << ", " << count[3] << "]" << endl; 
  return acc/ninst;
}

double Neighborhood::knn_test(MatrixXd& M, int kk){
  MatrixXd distMatrix(sd_test->ninst, sd->ninst);
  calcDistMatrix(distMatrix, M);
  double acc = .0;
  int count[4] = {0, 0, 0, 0};
  for(int i = 0; i < sd_test->ninst; ++ i){
    vector<DistPair> dp;
	for(int j = 0; j < sd->ninst; ++ j){
	  if(!inSameClass(sd_test->inst[i], sd->inst[j]) && !inOpposingClass(sd_test->inst[i], sd->inst[j]))
	    continue;
	  DistPair d = {j, distMatrix(i, j)};
	  dp.push_back(d);
	}
	/*
	if (i == 0)
	  cout << dp[0].dist << "," << dp[1].dist << "," << dp[2].dist << "---dist_knn(i=0)---" << endl;
	*/
	sort(dp.begin(), dp.end());
	/*
	if (i == 0)
	  cout << dp[0].ino << "," << dp[1].ino << "," << dp[2].ino << "---ino_knn(c++)---";
	if (i == sd_test->ninst - 1)
	  cout << dp[2].ino << endl;
	*/
	int similar_target_neighbor = 0;
	for(int j = 0; j < kk; ++ j){
	  if(inSameClass(sd_test->inst[i], sd->inst[dp[j].ino]))
	    ++ similar_target_neighbor;
	  //cout << dp[j].ino << ",";
	}
	if(similar_target_neighbor > kk/2){
	  acc += 1;
	  ++ count[sd_test->inst[i].label];
	}
  }
  //cout << "[" << count[0] << ", " << count[1] << ", " << count[2] << ", " << count[3] << "]" << endl; 
  return acc/sd_test->ninst;
}

