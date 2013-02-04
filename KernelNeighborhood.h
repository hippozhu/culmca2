
class KernelNeighborhood : public Neighborhood{
  public:
	KernelNeighborhood(SVMData&, SVMData&, int, int, int);
	
    MatrixXd K, K_test;
	vector<VectorXd> E;
	double gamma;
	
	struct VTriplet{
	  int i;
	  int j;
	  int l;
	  double vdist;
	  bool active;
	};
	vector<VTriplet> triplets;
	void initE();
	VectorXd& getE(int, int);
	virtual double dist(int, int, MatrixXd &);
	virtual double distance(int, int, MatrixXd&);
	
  private:
    void generateKernelMatrix();
	void generateTestKernelMatrix();
	double kernel_rbf(VectorXd&, VectorXd&);
    void initTriplets();
};
