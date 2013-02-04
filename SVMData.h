
#include <Eigen/Dense>
using namespace Eigen;
  
extern void deviceInitKernelMatrix(int *trainninst, int *testninst, int *nfeat, double *traindata, double *testdata);
extern void deviceInitTarget(int *h_target, int trainninst, int, int *, int *, int *, int *);
extern void deviceInitInstList(struct Inst *, unsigned *, unsigned, int, int, int);
extern void deviceInitMu(double, double);
extern void deviceInitO(double *, int);
extern void deviceInitTargetTerm(double *, int);
extern void deviceInitUpdateTerm(int, int);
extern void deviceInitTri(int);
extern void deviceInitLabelTrain(struct Inst *, unsigned);
extern void deviceInitLabelTest(struct Inst *, unsigned);
extern void kernelTest(int, int, int, int, double *, double, double, double, int);


/* ------------------------- SVM dataset class -------------------------------*/
class SVMData{
  public:
    SVMData(string);
    ~SVMData();
  
    int nfeat;
    int ninst;  
    Inst *inst;
	unsigned typecount[4];
    double *data, *data_col;
	
    void readData(const char*, bool, double *);
	double *getDataPoint(int);
	void getVector(int,VectorXd&);
	
  private:
	int getLineNo(const char*);
	int getNFeat(const char*);
	int stringToInt(string);
	double stringToFloat(string);
};
