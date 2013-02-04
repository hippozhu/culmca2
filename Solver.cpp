#include "mycommon.h"
#include "SVMData.h"
#include "Neighborhood.h"
#include "KernelNeighborhood.h"
#include "Solver.h"


Solver::Solver(KernelNeighborhood &neighborhood, double m, double a, double n):nb(&neighborhood), alpha(a), mu(m), nu(n) {
  initO();
  deviceInitMu(mu, nu);
  double *o = new double[(*nb).nfeat * (*nb).ninst];
  matrixToDouble(o, O);
  deviceInitO(o, O.size());
  computeTargetTerm();
  double *tt = new double[(*nb).ninst * (*nb).ninst];
  matrixToDouble(tt, t_target);
  deviceInitTargetTerm(tt, t_target.size());
  
  deviceInitUpdateTerm((*nb).ninst * (*nb).ninst, (*nb).nfeat * (*nb).ninst);
  
  deviceInitTri((*nb).ninst * (*nb).ninst);
}


bool Solver::converged(){
  return false;
}

void Solver::initO(){
  O = MatrixXd::Random((*nb).nfeat, (*nb).ninst);
}

void Solver::computeTargetTerm(){
  double s = 1.0;
  t_target = MatrixXd::Zero((*nb).ninst, (*nb).ninst);
  for(int i = 0; i < (*nb).ninst; i++)
	for(int j = 0; j < (*nb).nn[(*nb).sd->inst[i].label]; j++){
	  if ((*nb).sd->inst[i].label == TP)
	    s = nu;
	  else
	    s = 1.0;
	  int neighbor = (*nb).getTargetByOffset(i, j);
	  t_target.col(i) += s * (*nb).getE(i, (*nb).getTargetByOffset(i, j));
	  t_target.col((*nb).getTargetByOffset(i, j)) -= s * (*nb).getE(i, (*nb).getTargetByOffset(i, j));
	}
}

double Solver::hinge(double s){
  if (s <= -1.0)
    return .0;
  else if (s >= 0)
    return 1.0;
  else
    return 1 + s;
}

void Solver::updateTripletTerm(){
  t_triplet = MatrixXd::Zero((*nb).ninst, (*nb).ninst);
  for (vector<KernelNeighborhood::VTriplet>::iterator it = (*nb).triplets.begin(); it != (*nb).triplets.end(); it ++){
    if ((*it).active){
	
    double h = hinge((*it).vdist);
	if (h > 0){
	  t_triplet.col((*it).i) += h * (*nb).getE((*it).l, (*it).j);
	  t_triplet.col((*it).j) += h * (*nb).getE((*it).j, (*it).i);
	  t_triplet.col((*it).l) += h * (*nb).getE((*it).i, (*it).l);
	}
	
	}
  }
}

void Solver::updateVdist(){
  for (vector<KernelNeighborhood::VTriplet>::iterator it = (*nb).triplets.begin(); it != (*nb).triplets.end(); it ++){
    if ((*it).active)
	  (*it).vdist = (*nb).violatedDist((*it).i, (*it).j, (*it).l, O);
  }
}

void Solver::updateActiveSet(){
  int idx = 0;
  for (vector<KernelNeighborhood::VTriplet>::iterator it = (*nb).triplets.begin(); it != (*nb).triplets.end(); it ++){
    if (!(*it).active){
	
	  (*it).vdist = (*nb).violatedDist((*it).i, (*it).j, (*it).l, O);
	  if((*it).vdist > -1){
	    (*it).active = true;
		active_set.push_back(idx);
	  }
	}
	++ idx;
	if (idx % 100000 == 0)
	  cout << "au: " << idx << endl;
  }
  //cout << "active size: " << active_set.size() << endl;
}

double Solver::objFunc(){
  double f = .0;
  for(int i = 0; i < (*nb).ninst; ++ i)
	for(int j = 0; j < (*nb).k; ++ j){
	  double tmp = (*nb).dist(i, (*nb).getTarget(i, j), O);
	  f += (*nb).dist(i, (*nb).getTarget(i, j), O);
	}
  //cout << endl << "obj target part: " << f << endl;
	
  for (vector<KernelNeighborhood::VTriplet>::iterator it = (*nb).triplets.begin(); it != (*nb).triplets.end(); it ++){
    if ((*it).vdist > .0)
	  f += (*it).vdist;
  }
  //cout << "obj final part: " << f << endl;
  
  return f;
}

void Solver::solve(){
  cout << "initializing f ... " << endl;
  initO();
  computeTargetTerm();
  
  cout << "initial alpha=" << alpha << endl;
  updateActiveSet();
  //updateActiveSet_thread();
  f_val = objFunc();
  double acc = (*nb).knn(O, (*nb).k, true);
  double acc_test;
  cout << "initial f = " << f_val << ", knn = "<< acc << endl;
  
  unsigned iter = 0;
  clock_t t0;
  bool reduced = true;
  
  while(!converged()){
	++ iter;
	if (iter % 10 == 0 || !reduced){
	  acc = (*nb).knn(O, (*nb).k, false);
	  acc_test = (*nb).knn_test(O, (*nb).k);
	  cout << "knn=" << acc << ", knn_test=" << acc_test << endl;
	  cout << "UPDATE at iter = " << iter << "..." << endl;
	  updateActiveSet();
	  //updateActiveSet_thread();
	}
    t0 = clock();
    updateTripletTerm();
	cout << "updateTripletTerm: " << 1.0*(clock() - t0)/CLOCKS_PER_SEC << endl << t_triplet(0, 0) << " " << t_triplet(1, 0) << " " << t_triplet(2, 0) << " " << t_triplet(3, 0) << " " << t_triplet(4, 0) << " " << t_triplet(5, 0) << endl;
	t0 = clock();
    G = 2 * O * (t_target + mu * t_triplet);
    O -= alpha * G;
	cout << "update O: " << 1.0*(clock() - t0)/CLOCKS_PER_SEC << endl;
	t0 = clock();
	
	updateVdist();
	cout << "updateVdist:" << 1.0*(clock() - t0)/CLOCKS_PER_SEC << endl;
	t0 = clock();
	double f = objFunc();
	if (f < f_val){
	  alpha *= 1.05;
	  reduced = true;
	}
	else{
	  alpha /= 2;
	  reduced = false;
	}
	f_val = f;
	cout << "iter " << iter << ": f= " << f_val << ", alpha =" << alpha << endl;
  }
}


void Solver::matrixToDouble(double *d, MatrixXd& m){
  for (int i = 0; i < m.size(); ++ i)
    d[i] = m(i/m.cols(), i%m.cols());
}
/*
void Solver::doubleToMatrix(){
  int size = O.size();
  for (int i = 0; i < size; ++ i)
    O(i/O.cols(), i%O.cols()) = o[i];
}
*/
void Solver::cusolve(){
  //cout << "initializing f ... " << endl;
  //cout << "initial alpha=" << alpha << endl;
  
  cout << endl;
  //updateActiveSet();
  //f_val = objFunc();
  double acc;
  double acc_test;
  //cout << "initial f = " << f_val << endl;
  //cout << "knn = "<< acc << ", test knn = "<< acc_test << endl;
  
  unsigned iter = 0;
  clock_t t0;
  bool reduced = true;
  f_val = DBL_MAX;
  
  while(!converged()){
    cout << endl << "Iter " << iter << ", alpha =" << alpha << endl;
    t0 = clock();
	++ iter;
	/*
	if (iter % 1 == 0 || !reduced){
	  acc = (*nb).knn(O, (*nb).k, false);
	  acc_test = (*nb).knn_test(O, (*nb).k);
	  cout << "-------------knn=" << acc << ", knn_test=" << acc_test << endl;
	  //cout << "UPDATE at iter = " << iter << "..." << endl;
	  updateActiveSet();
	}*/
    updateActiveSet();
	
	double f = objFunc();
	if (f < f_val){
	  alpha *= 1.05;
	  reduced = true;
	}
	else{
	  alpha /= 2;
	  reduced = false;
	}
	f_val = f;
	
	acc = (*nb).knn(O, (*nb).k, true);
	acc_test = (*nb).knn_test(O, (*nb).k);
	//cout << endl;
    cout <<"f_val = " << f_val << ", knn = "<< acc << ", knn_test = "<< acc_test << endl;
	//cout << "updating O ..." << endl;
    updateTripletTerm();
	//cout << t_triplet(0, 0) << "," << t_triplet(0, 1) << "," << t_triplet(0, 2) << "---t_triplet---" << t_triplet(306, 306) << endl;
	//cout << t_target(0, 0) << "," << t_target(0, 1) << "-" << t_target(0, 2) << "---t_target---" << t_target(306, 306) << endl;
	
	MatrixXd u = MatrixXd::Identity((*nb).ninst, (*nb).ninst) - 2 * alpha * (t_target + mu * t_triplet);
	//cout << u(0, 0) << "," << u(0, 1) << "," << u(0, 2) << "---t_update---" << u((*nb).ninst - 1, (*nb).ninst - 1) << endl;
	//cout << "row 0 of O:" << endl << O.row(0) << endl;
	//cout << "col 0 of U:" << u.col(512) << endl;
	O *= u;
	//cout << "c++ 512: " << O(0, 512) << endl;

    //G = 2 * O * (t_target + mu * t_triplet);
    //O -= alpha * G;
	//cout << "update O: " << 1.0*(clock() - t0)/CLOCKS_PER_SEC << endl;
	//cout << O(0, 0) << "," << O(0, 1) << "," << O(0, 2) << "---C++ O---" << O((*nb).nfeat - 1, (*nb).ninst - 1) << endl;
	/*
	for (int i = 0; i < (*nb).nfeat; ++ i)
	  for (int j = 0; j < (*nb).ninst; ++ j)
	    cout << O(i, j) << ",";
	cout << endl << endl;
	*/
	
	updateVdist();
	//cout << "updateVdist:" << 1.0*(clock() - t0)/CLOCKS_PER_SEC << endl;
	cout << "time:" << 1.0*(clock() - t0)/CLOCKS_PER_SEC << endl;
	if (iter > 10)
	  break;
  }
}
/*
void Solver::initIsActive(){
  for(int i = 0; i < (*nb).ninst*(*nb).ninst*(*nb).ninst; i++)
    in_active.push_back(false);
}

bool Solver::isActive(int i, int j, int l){
  return in_active[i * (*nb).ninst * (*nb).ninst + j * (*nb).ninst + l];
}

void Solver::setActive(int i, int j, int l){
  in_active[i * (*nb).ninst * (*nb).ninst + j * (*nb).ninst + l] = true;
}

void Solver::initC(){
  for(int i = 0; i < (*nb).ninst*(*nb).ninst; i ++){
    MatrixXd v;
	C.push_back(v);
  }
}

MatrixXd Solver::getC(int i, int j){
  if(C[i * (*nb).ninst + j].rows()==0)
    C[i * (*nb).ninst + j] = (*nb).outerProduct(i, j);
  return C[i * (*nb).ninst + j];
}

void Solver::initG(){
  G = MatrixXd::Zero((*nb).nfeat, (*nb).nfeat);
  for(int i = 0; i < (*nb).nfeat; i++)
	for(int j = 0; j < (*nb).k; j++)
	  if (flavored(i))
	    G += times * (1 - mu) * getC(i, (*nb).getTarget(i, j));
	  else
	    G += (1 - mu) * getC(i, (*nb).getTarget(i, j));
}

void Solver::updateG(){
  nonv += nonviolated.size();
  v += violated.size();
  cout << "G:" << nonviolated.size() << "," << violated.size() << " ";
  for(vector<int>::iterator it = nonviolated.begin(); it != nonviolated.end(); it ++){
    VTriplet vt = triplets[*it];
	if (flavored(vt.i))
	  G -= times * mu * (getC(vt.i, vt.j) - getC(vt.i, vt.l));
	else
	  G -= mu * (getC(vt.i, vt.j) - getC(vt.i, vt.l));
  }
  
  for(vector<int>::iterator it = violated.begin(); it != violated.end(); it ++){
    VTriplet vt = triplets[*it];
	if (flavored(vt.i))
	  G += times * mu * (getC(vt.i, vt.j) - getC(vt.i, vt.l));
	else
	  G += mu * (getC(vt.i, vt.j) - getC(vt.i, vt.l));
  }
}

void Solver::updateM(){
  //cout << "G: with alpha=" << alpha << endl << G << endl << endl;
  //cout << "M:" << endl << M << endl << endl;
  // << "D:" << endl << D << endl << endl;
  //cout << "after D:" << endl << D << endl << endl; 
  //cout << "M:" << endl << M << endl << endl; 
  //cout << "V:" << endl << V << endl << endl;  
  M -= alpha * G;  
  //cout << "M:" << endl << M << endl << endl;  
  EigenSolver<MatrixXd> es(M);
  MatrixXd D = es.pseudoEigenvalueMatrix();
  MatrixXd V = es.pseudoEigenvectors(); 
  //cout << "D:" << endl << D << endl << endl;
  int projected = 0;  
  for (int i = 0; i < (*nb).nfeat; i ++)
    if (D(i, i) < 1e-10){
	  D(i, i) = .0;
	  ++ projected;
	}
  if (projected > 0){
    M = V * D * V.transpose();
    cout << "M updated:" << projected << endl;
  }
}

void Solver::updateVdist(){
  violated.clear();
  nonviolated.clear();
  for(vector<int>::iterator it = activeSet.begin(); it != activeSet.end(); it ++){
    triplets[*it].vdist = (*nb).violatedDist(triplets[*it].i, triplets[*it].j, triplets[*it].l, M);
	if (triplets[*it].vdist < 0 && triplets[*it].violated){
      nonviolated.push_back(*it);
	  triplets[*it].violated = false;
	}
	else if (triplets[*it].vdist > 0 && !triplets[*it].violated){
	  violated.push_back(*it);
	  triplets[*it].violated = true;
	}
	else
	  ;
  }
}

void Solver::updateOthers(){
  double new_fval = evalObjFunc();
  reduced = new_fval < f_val;
  if(new_fval > f_val)
    alpha *= 0.5;
  else if(new_fval < f_val)
    alpha *= 1.01;
  else
    ;
  f_val = new_fval;
}

bool Solver::flavored(int i){
  //return (*nb).isType(i, Neighborhood::TN) || (*nb).isType(i, Neighborhood::TP);
  //return (*nb).isType(i, Neighborhood::FN) || (*nb).isType(i, Neighborhood::FP);
  //return (*nb).isType(i, Neighborhood::TN) || (*nb).isType(i, Neighborhood::FN);
  //return (*nb).isType(i, Neighborhood::TN);
  //return (*nb).isType(i, Neighborhood::TP);
  //return (*nb).isType(i, Neighborhood::FP);
  //return (*nb).isType(i, Neighborhood::FN);
  //return false;
  return (*nb).isType(i, Neighborhood::TP);
}

double Solver::evalObjFunc(){
  double fval = .0;
  for(int i = 0; i < (*nb).ninst; i++)
    for(int j = 0; j < (*nb).k; j ++)
	  if (flavored(i))
	    fval += times * (1 - mu) * (*nb).mdist(i, (*nb).getTarget(i, j), M);
	  else
	    fval += (1 - mu) * (*nb).mdist(i, (*nb).getTarget(i, j), M);
  for(vector<int>::iterator it = activeSet.begin(); it != activeSet.end(); it ++)
    if(triplets[*it].vdist > 0)
	  if (flavored(triplets[*it].i))
        fval += times * mu * triplets[*it].vdist;
	  else
        fval += mu * triplets[*it].vdist;

  return fval;
}

void Solver::activeSetUpdate(){
  int idx = 0;
  for(vector<VTriplet>::iterator it = triplets.begin(); it != triplets.end(); it ++){
    if(!(*it).active){
	  double vd = (*nb).violatedDist((*it).i, (*it).j, (*it).l, M);
	  //cout << "active M:" << endl << M << endl;
	  if(vd > 0){
	    //cout << "active added" << endl;
		  (*it).vdist = vd;
		  (*it).active = true;
		  (*it).violated = true;
		  activeSet.push_back(idx);
	      violated.push_back(idx);
	  }
    }	
	++ idx;
  }
}


void Solver::solve(){
  initC();
  initG();
  initTriplets();
  reduced = true;
  int t = 0;
  cout << "activeSetUpdate() ..." << endl;
  activeSetUpdate();
  cout << "evalObjFunc() ..." << endl;
  f_val = evalObjFunc();
  double acc = (*nb).knnWeighted(M, (*nb).k);
  //double acc = (*nb).knn(M, 3);
  cout << "acd=" << (*nb).acd << endl;
  cout << "Initial f_val=" << f_val << ", knn=" << acc << endl;
	  clock_t t0 = clock();
  while(!converged()){
    ++ t;
	if (t % 10 == 0 || !reduced){
	  activeSetUpdate();
	  cout << "knn() ..." << endl;
	  acc = (*nb).knnWeighted(M, (*nb).k);
	  //acc = (*nb).knn(M, 3);
	  cout << "iter: " << t << "  t1: " << 1.0*(clock() - t0)/CLOCKS_PER_SEC << ", step=" << alpha 
	  << "(v - nonv)= " << v << "-" << nonv << "=" << v-nonv << endl;
	  cout << "active=" << activeSet.size() << ", f_val=" << f_val<< ", knn=" << acc << endl;
	  t0 = clock();
	}
	//violateSetUpdate();
	updateG();
	updateM();
	updateVdist();
	updateOthers();
	if(t==100)
	  break;
  }
}

void Solver::updateVdist_thread(){
  int active_size = active_set.size();
  int chunk = active_size/NThread;
  
  boost::thread_group tg;  
  for(int i = 0; i < NThread; ++ i)
    tg.create_thread(boost::bind(&Solver::partialVdist, this, i * chunk, min((i + 1) * chunk, active_size)));
  tg.join_all();
}

void Solver::partialVdist(int start, int end){
  for (int i = start; i < end; ++ i){
    int active_idx = active_set[i];
    (*nb).triplets[active_idx].vdist = (*nb).violatedDist((*nb).triplets[active_idx].i, (*nb).triplets[active_idx].j, (*nb).triplets[active_idx].l, O);
  }
}

void Solver::updateActiveSet_thread(){
  cout << "active set main" << endl;
  int triplet_size = (*nb).triplets.size();
  int chunk = triplet_size/NThread;
  boost::thread_group tg;  
  for(int i = 0; i < NThread; ++ i)
    tg.create_thread(boost::bind(&Solver::partialUpdateActiveSet, this, i, i * chunk, min((i + 1) * chunk, triplet_size)));
  tg.join_all();
}

//void Solver::partialUpdateActiveSet(int tid){
void Solver::partialUpdateActiveSet(int tid, int start, int end){
  //cout << "active set thread " << tid << ", start end: " << start << " " << end << endl;
  int count = 0;
  //for(int i = tid; i < (*nb).triplets.size(); i += NThread){
  for(int i = start; i < end; ++ i){
    if (!(*nb).triplets[i].active){
	  double d = (*nb).violatedDist((*nb).triplets[i].i, (*nb).triplets[i].j, (*nb).triplets[i].l, O);
	  //boost::mutex::scoped_lock lock(udpate_mutex);
	  (*nb).triplets[i].vdist = d;
	  if((*nb).triplets[i].vdist > -1){
	    (*nb).triplets[i].active = true;
		active_set.push_back(i);
	  }
	  
	}
	++ count;
	if (count % 10000 == 0){
	  boost::mutex::scoped_lock lock(Solver::udpate_mutex);
	  cout << "active set udpate thread " << tid << " complete " << count << endl;
	}
  }
}
*/

