#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#define BOOST_UBLAS_NESTED_CLASS_DR45

#include <boost/lexical_cast.hpp>
#include <boost/math/special_functions/sinc.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>
#include <kml/gaussian.hpp>
#include <kml/svm.hpp>
#include <kml/io.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <utility>

using std::string; using std::cout; using std::endl; 
using std::ifstream; using std::stringstream;
using std::getline; using std::cerr;

namespace ublas = boost::numeric::ublas;

typedef kml::ranking<std::vector<double>, int> problem_type;
typedef std::vector<std::vector<double> >::iterator vec_iter;

int main(int argc, char *argv[]) {

  std::vector<std::vector<double> > points;
  std::vector<double> point;
  point.push_back(1); point.push_back(1); point.push_back(0); point.push_back(0.2); point.push_back(0);
  points.push_back(point);
  point.resize(0);
  point.push_back(0); point.push_back(0); point.push_back(1); point.push_back(0.1); point.push_back(1);
  points.push_back(point);
  point.resize(0);
  point.push_back(0); point.push_back(1); point.push_back(0); point.push_back(0.4); point.push_back(0);
  points.push_back(point);
  point.resize(0);
  point.push_back(0); point.push_back(0); point.push_back(1); point.push_back(0.3); point.push_back(0);
  points.push_back(point);
  point.resize(0);

  point.push_back(0); point.push_back(0); point.push_back(1); point.push_back(0.2); point.push_back(0);
  points.push_back(point);
  point.resize(0);
  point.push_back(1); point.push_back(0); point.push_back(1); point.push_back(0.4); point.push_back(0);
  points.push_back(point);
  point.resize(0);
  point.push_back(0); point.push_back(0); point.push_back(1); point.push_back(0.1); point.push_back(0);
  points.push_back(point);
  point.resize(0);
  point.push_back(0); point.push_back(0); point.push_back(1); point.push_back(0.2); point.push_back(0);
  points.push_back(point);
  point.resize(0);

  point.push_back(0); point.push_back(0); point.push_back(1); point.push_back(0.1); point.push_back(1);
  points.push_back(point);
  point.resize(0);
  point.push_back(1); point.push_back(1); point.push_back(0); point.push_back(0.3); point.push_back(0);
  points.push_back(point);
  point.resize(0);
  point.push_back(1); point.push_back(0); point.push_back(0); point.push_back(0.4); point.push_back(1);
  points.push_back(point);
  point.resize(0);
  point.push_back(0); point.push_back(1); point.push_back(1); point.push_back(0.5); point.push_back(0);
  points.push_back(point);
  point.resize(0);

  std::vector<std::pair<int, int> > target;
  target.push_back(std::make_pair(1,3));
  target.push_back(std::make_pair(1,2));
  target.push_back(std::make_pair(1,1));
  target.push_back(std::make_pair(1,1));
  target.push_back(std::make_pair(2,1));
  target.push_back(std::make_pair(2,2));
  target.push_back(std::make_pair(2,1));
  target.push_back(std::make_pair(2,1));
  target.push_back(std::make_pair(3,2));
  target.push_back(std::make_pair(3,3));
  target.push_back(std::make_pair(3,4));
  target.push_back(std::make_pair(3,1));

  kml::svm<problem_type, kml::gaussian> my_machine(3.162277, 1.0);
  my_machine.learn(points, target);

  std::cerr << "Weight vector (size " << my_machine.weight.size() << "): ";
  for (unsigned int i=0; i<my_machine.weight.size(); ++i)
    std::cerr << my_machine.weight[i] << ", ";
  std::cerr << std::endl;

  for (std::vector<std::vector<double> >::const_iterator it1 = 
	 my_machine.support_vector.begin();
       it1 != my_machine.support_vector.end(); ++it1) {
    std::cerr << "Vector size: " << it1->size() << std::endl;
    std::cerr << "[";
    for (std::vector<double>::const_iterator it2 = it1->begin();
	 it2 != it1->end(); ++it2)
      std::cerr << *it2 << ", ";
    std::cerr << "]" << std::endl;
  }

  std::vector<std::vector<double> > testpoints;
  point.push_back(1);
  point.push_back(0);
  point.push_back(0);
  point.push_back(0.2);
  point.push_back(1);
  testpoints.push_back(point);
  point.resize(0);
  point.push_back(1);
  point.push_back(1);
  point.push_back(0);
  point.push_back(0.3);
  point.push_back(0);
  testpoints.push_back(point);
  point.resize(0);
  point.push_back(0);
  point.push_back(0);
  point.push_back(0);
  point.push_back(0.2);
  point.push_back(1);
  testpoints.push_back(point);
  point.resize(0);
  point.push_back(0);
  point.push_back(0);
  point.push_back(1);
  point.push_back(0.2);
  point.push_back(0);
  testpoints.push_back(point);
  point.resize(0);

  std::vector<int> testtarget;

  for (vec_iter i = testpoints.begin(); i != testpoints.end(); ++i)
    cout << my_machine(*i) << endl;
  return 0;
}
