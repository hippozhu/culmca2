#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <math.h>
#include <float.h>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <iterator>
#include <limits>
#include <cmath>
#include <cstdlib>

#define TN 0
#define TP 1
#define FP 2
#define FN 3

using namespace std;
	
	
struct Inst{
  unsigned short ino;
  short label;//class label (+1/-1)
  //short int status;
  //short int confusion;
};