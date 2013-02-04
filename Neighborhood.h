
/* ------------------------- Neighborhood class -------------------------------*/
class Neighborhood{
  public:
    Neighborhood(SVMData&, SVMData&, int, int, int);
	~Neighborhood();
    SVMData *sd, *sd_test;
	int nclass;
	int k;
	int nn[4];
	int *target_offset;
	int target_size;
	
    int nfeat;
    int ninst;
    int ninst_test;
	int *target;
	double acd; // average closest distance
	double e_acc; // knn accuracy by euclidean distance
	
	struct DistPair{
	  int ino;
	  double dist;
	  
		bool operator<(const DistPair& a) const
		{
			return dist < a.dist;
		}
	};
	
	bool inSameClass(int, int);
	bool inOpposingClass(int, int);
	bool inSameClass(Inst&, Inst&);
	bool inOpposingClass(Inst&, Inst&);
	bool isType(int, int);
	double mdist(int, int, MatrixXd &);
	void findTarget();
	double violatedDist(int, int, int, MatrixXd &);
	int getTarget(int, int);
	int getTargetByOffset(int, int);
	double averageClosestDist();
	MatrixXd outerProduct(int, int);
	double knn(MatrixXd&, int, bool);
	double knn_test(MatrixXd&, int);
	virtual double dist(int, int, MatrixXd &);
	virtual double distance(int, int, MatrixXd&);
	
	
  private:
    void calcEdistMatrix(double *);
    void calcDistMatrix(double *, MatrixXd &);
    void calcDistMatrix(MatrixXd &, MatrixXd &);
	double edist(int, int);
	void dataPointToVector(double *p, VectorXd &);
	double weight(double);
};