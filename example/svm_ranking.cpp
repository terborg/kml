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
#include <kml/ranking.hpp>
#include <kml/svm.hpp>
#include <kml/io.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/vector_property_map.hpp>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

using std::string; using std::cout; using std::endl; 
using std::ifstream; using std::stringstream;
using std::getline; using std::cerr;

namespace ublas = boost::numeric::ublas;

typedef boost::tuple<std::vector<double>, int, double> example_type;
typedef kml::ranking<example_type> problem_type;
typedef kml::gaussian<problem_type::input_type> kernel_type;
typedef boost::vector_property_map<example_type> PropertyMap;

int main(int argc, char *argv[]) {

  PropertyMap data;
  problem_type::input_type point;

  point.push_back(1); point.push_back(1); point.push_back(0); point.push_back(0.2); point.push_back(0);
  data[0] = boost::make_tuple(point, 1, 3);
  point.resize(0);
  point.push_back(0); point.push_back(0); point.push_back(1); point.push_back(0.1); point.push_back(1);
  data[1] = boost::make_tuple(point, 1, 2);
  point.resize(0);
  point.push_back(0); point.push_back(1); point.push_back(0); point.push_back(0.4); point.push_back(0);
  data[2] = boost::make_tuple(point, 1, 1);
  point.resize(0);
  point.push_back(0); point.push_back(0); point.push_back(1); point.push_back(0.3); point.push_back(0);
  data[3] = boost::make_tuple(point, 1, 1);
  point.resize(0);

  point.push_back(0); point.push_back(0); point.push_back(1); point.push_back(0.2); point.push_back(0);
  data[4] = boost::make_tuple(point, 2, 1);
  point.resize(0);
  point.push_back(1); point.push_back(0); point.push_back(1); point.push_back(0.4); point.push_back(0);
  data[5] = boost::make_tuple(point, 2, 2);
  point.resize(0);
  point.push_back(0); point.push_back(0); point.push_back(1); point.push_back(0.1); point.push_back(0);
  data[6] = boost::make_tuple(point, 2, 1);
  point.resize(0);
  point.push_back(0); point.push_back(0); point.push_back(1); point.push_back(0.2); point.push_back(0);
  data[7] = boost::make_tuple(point, 2, 1);
  point.resize(0);

  point.push_back(0); point.push_back(0); point.push_back(1); point.push_back(0.1); point.push_back(1);
  data[8] = boost::make_tuple(point, 3, 2);
  point.resize(0);
  point.push_back(1); point.push_back(1); point.push_back(0); point.push_back(0.3); point.push_back(0);
  data[9] = boost::make_tuple(point, 3, 3);
  point.resize(0);
  point.push_back(1); point.push_back(0); point.push_back(0); point.push_back(0.4); point.push_back(1);
  data[10] = boost::make_tuple(point, 3, 4);
  point.resize(0);
  point.push_back(0); point.push_back(1); point.push_back(1); point.push_back(0.5); point.push_back(0);
  data[11] = boost::make_tuple(point, 3, 1);
  point.resize(0);

  kml::svm<problem_type, kernel_type, PropertyMap> my_machine(3.162277, 1.0, data);
  my_machine.learn(data.storage_begin(), data.storage_end());

  /*
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
  */
  std::vector<problem_type::input_type> testpoints;
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

  for (std::vector<problem_type::input_type>::iterator i = testpoints.begin(); i != testpoints.end(); ++i)
    cout << my_machine(*i) << endl;
  return 0;
}
