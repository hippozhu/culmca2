
/* ------------------------- Solver class -------------------------------*/
class Solver{
  public:
    Solver(KernelNeighborhood &, double, double, double);
	
	KernelNeighborhood *nb;
    MatrixXd O;
	MatrixXd t_target;
	MatrixXd t_triplet;
	MatrixXd G;
	double alpha;  // step size
	double mu;
	double nu;
	double f_val;
	void solve();
	void cusolve();
	vector<int> active_set;
	//int NThread;
	//void updateVdist_thread();
	//void partialVdist(int start, int end);
	//void updateActiveSet_thread();
	//void partialUpdateActiveSet(int);
	//void partialUpdateActiveSet(int, int, int);
	//static boost::mutex udpate_mutex;

  private:
	double evalObjFunc();
	void initO();
	bool converged();
	void updateActiveSet();
	void computeGamma();
	void computeTargetTerm();
	void updateTripletTerm();
	double hinge(double);
	void updateVdist();
	double objFunc();
	void matrixToDouble(double *, MatrixXd&);
	void doubleToMatrix();
};