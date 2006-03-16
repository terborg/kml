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
#include <boost/tuple/tuple.hpp>
#include <boost/vector_property_map.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <map>

#include <boost/static_assert.hpp>
#include <boost/type_traits/is_same.hpp>

using std::string; using std::cout; using std::endl; 
using std::ifstream; using std::stringstream;
using std::getline; using std::cerr;

namespace ublas = boost::numeric::ublas;


int main(int argc, char *argv[]) {

  typedef boost::tuple<std::vector<double>, bool> example_type;
  typedef boost::tuples::element<0, example_type>::type input_type;
  typedef kml::classification<example_type> problem_type;
  typedef boost::vector_property_map<example_type> data_type;
  typedef std::vector<double>::iterator vec_iter;
  typedef kml::gaussian<problem_type::input_type> kernel_type;

  if (argc < 2) {
    cout << "Error: need an input file to train and test on." << endl 
	      << "usage: svm_classification (trainfile) (testfile)" << endl;
    return 0;
  }

  BOOST_STATIC_ASSERT((boost::is_same<boost::property_traits<data_type>::value_type, example_type >::type::value));

  data_type data;
  std::vector<input_type> learn_keys;
  cerr << "Reading in vector...";
  kml::file train_file(argv[1]);
  train_file.read(data, learn_keys);
  
  cerr << "Done, " << std::distance(learn_keys.begin(), learn_keys.end()) << " points of size " << learn_keys[0].size() << endl;
  kml::svm<problem_type, kernel_type, data_type> my_machine(3.162277, 1.0, data);


  my_machine.learn(learn_keys.begin(), learn_keys.end());
  cerr << "Done training" << endl;
  for (std::vector<double>::iterator i = my_machine.weight.begin();
       i != my_machine.weight.end(); ++i)
    cout << boost::lexical_cast<double>(*i) << " ";
  cout << endl;

  data_type test_data;
  std::vector<input_type> test_keys;
  std::vector<input_type>::iterator i;
  cerr << "Reading in test vector...";
  kml::file test_file(argv[2]);
  cerr << "Done, " << std::distance(test_keys.begin(), test_keys.end()) << " points of size " << test_keys[0].size() << endl;
  for (i = test_keys.begin(); i != test_keys.end(); ++i)
    cout << my_machine(*i) << endl;
  return 0;
}
