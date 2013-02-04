#include "mycommon.h"
#include "SVMData.h"

SVMData::SVMData(string datafile){
  nfeat = getNFeat(datafile.c_str());
  ninst = getLineNo(datafile.c_str());
  data = new double[nfeat*ninst];
  data_col = new double[nfeat*ninst];
  inst = new Inst[ninst];
  readData(datafile.c_str(), false, data);
  readData(datafile.c_str(), true, data_col);
}

SVMData::~SVMData(){
  free(data);
  free(inst);
}

int SVMData::getLineNo(const char* filename){
  ifstream myfile(filename);
  if (!myfile.is_open()){
    cout << "Unable to open file: " << filename << endl;
	exit (EXIT_FAILURE);
  }
  myfile.unsetf(ios_base::skipws);
  int lineno = count(istream_iterator<char>(myfile), istream_iterator<char>(), '\n');
  myfile.close();
  return lineno;
}

int SVMData::getNFeat(const char* filename){
  ifstream myfile (filename);
  int nf = -1;
  string line;
  if (myfile.is_open()){
    getline(myfile,line);
    int pspace = line.find_first_of(" ");
	istringstream iss(line.substr(pspace+1));
      do{
        string sub;
        iss >> sub;
        if (sub.size()==0)
          break;
		int pcolon = sub.find_first_of(":");
		nf = stringToInt(sub.substr(0, pcolon));
      } while (iss); 
	  
  }
  else{
    cout << "Unable to open file: " << filename << endl;
	exit (EXIT_FAILURE);
  }
  return nf;
}

void SVMData::readData(const char* filename, bool columnwise, double *d){

  if (!columnwise){
    typecount[TN] = typecount[TP] = typecount[FP] = typecount[FN] = 0;
  }
  
  for (int i=0; i<ninst*nfeat; i++)
    d[i] = 0.0;
  string line;
  int lineno = 0;
  ifstream myfile (filename);
  if (myfile.is_open())
  {
    while ( myfile.good() )
    {
      getline (myfile,line);
      if (line.size()==0)
        break;
      int pspace = line.find_first_of(" ");
	  
	  if(!columnwise){
        inst[lineno].ino = lineno;
        //inst[lineno].status = 0;
        inst[lineno].label = stringToInt(line.substr(0, pspace));
	    ++ typecount[inst[lineno].label];
	  }

      istringstream iss(line.substr(pspace+1));
      do{
        string sub;
        iss >> sub;
        if (sub.size()==0)
          break;
		int pcolon = sub.find_first_of(":");
		int idxFeat = stringToInt(sub.substr(0, pcolon))-1;
		double f = stringToFloat(sub.substr(pcolon+1));
		if (columnwise)
		  d[lineno + idxFeat*ninst] = f;
		else
		  d[lineno*nfeat + idxFeat] = f;
      } while (iss); 

      lineno ++;
    }
    myfile.close();
  }
  else{
    cout << "Unable to open file: " << filename << endl;
	exit (EXIT_FAILURE);
  }
}

double* SVMData::getDataPoint(int i){
  return data+i*nfeat;
}

void SVMData::getVector(int i, VectorXd& v){
  for(int j = 0; j < nfeat; j++)
    v(j) = (data+i*nfeat)[j];
}

int SVMData::stringToInt(string s){
  int result;
  stringstream ss(s);
  ss >> result;
  return result;
}

double SVMData::stringToFloat(string s){
  double result;
  stringstream ss(s);
  ss >> result;
  return result;
}
