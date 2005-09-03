/*****************************************************************************
 *  The Kernel-Machine Library                                               *
 *  Copyright (C) 2004, 2005 by Rutger W. ter Borg and Meredith L. Patterson *
 *                                                                           *
 *  This program is free software; you can redistribute it and/or            *
 *  modify it under the terms of the GNU General Public License              *
 *  as published by the Free Software Foundation; either version 2           *
 *  of the License, or (at your option) any later version.                   *
 *                                                                           *
 *  This program is distributed in the hope that it will be useful,          *
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of           *
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the            *
 *  GNU General Public License for more details.                             *
 *                                                                           *
 *  You should have received a copy of the GNU General Public License        *
 *  along with this program; if not, write to the Free Software              *
 *  Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307    *
 *****************************************************************************/

/*****************************************************************************
 * Department of Credit Where Credit is Due:                                 *
 * This program uses the Sequential Minimal Optimization (SMO) algorithm,    *
 * originally developed by John Platt. You can find the pseudocode for it at *
 * http://research.microsoft.com/users/jplatt/smo.html.                      *
 *                                                                           *
 * The ranking SVM uses a technique developed by Thorsten Joachims in        *
 * "Optimizing Search Engines Using Clickthrough Data," Proceedings of the   *
 * ACM Conference on Knowledge Discovery and Data Mining (KDD), ACM, 2002.   *
 *****************************************************************************/

#ifndef SVM_HPP
#define SVM_HPP

#define EPS .001

#include <boost/numeric/bindings/traits/std_vector.hpp>
#include <boost/numeric/bindings/traits/ublas_vector.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/mpl/equal_to.hpp>
#include <boost/type_traits/is_float.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/random_number_generator.hpp>
#include <boost/range/value_type.hpp>

#include <algorithm>
#include <vector>
#include <cstdlib>
#include <ext/functional>
#include <iostream>

#include <kml/determinate.hpp>
#include <kml/classification.hpp>
#include <kml/ranking.hpp>
#include <kml/regression.hpp>


namespace kml {

  template<typename Problem, template<typename, int> class K, class Enable = void>
  class svm: public determinate<typename Problem::input_type, typename Problem::output_type, K> {};

  // Classification SVM

  template<typename Problem, template<typename, int> class K>
  class svm<Problem, K, typename boost::enable_if<is_classification<Problem> >::type >:
    public determinate<typename Problem::input_type, typename Problem::output_type, K> {
public:
  typedef determinate<typename Problem::input_type, typename Problem::output_type, K> base_type;
  typedef typename base_type::kernel_type kernel_type;
  typedef typename base_type::result_type result_type;
  typedef typename Problem::input_type input_type; // I wonder if I need that extra ::type
  typedef typename Problem::output_type output_type;  
  typedef double scalar_type;

  // Bug fixed: use call_traits instead of const&, so can be called with both reference and non-reference
  svm( typename boost::call_traits<kernel_type>::param_type k,
       typename boost::call_traits<double>::param_type max_weight ): 
    base_type(k), C(max_weight), tol(.001), startpt(randomness) { }

  result_type operator() (input_type const &x) {
    result_type ret=0;
    for (size_t i=0; i<base_type::weight.size(); ++i)
      if (base_type::weight[i] > 0) 
	ret += base_type::weight[i] * target[i] * base_type::kernel(points[i], x);
    ret -= base_type::bias;
    return ret;
  }

  int takeStep(int i1, int i2) {
    if (i1 == i2) return 0; 

    double alpha1 = base_type::weight[i1];
    output_type y1 = target[i1];
    scalar_type e1, e2, L, H, a2;
    /* p. 49, Platt: "When an error E is required by SMO, it will look up the error in the error cache if the 
       corresponding Lagrange multiplier is not at bound." */
    if (0 != alpha1 && C != alpha1) 
      e1 = error_cache[i1];
    else
      e1 = operator()(points[i1]) - y1;

    double alpha2 = base_type::weight[i2];
    output_type y2 = target[i2];
    /* p. 49 again */
    if (0 != alpha2 && C != alpha2)
      e2 = error_cache[i2];
    else
      e2 = (double)operator()(points[i2]) - y2;
    
    output_type s = y1 * y2;

    if (y1 != y2) {
      L = std::max(0.0, alpha2 - alpha1);
      H = std::min(C, C + alpha2 - alpha1);
    }
    else {
      /* Equation 12.3 */
      L = std::max(0.0, alpha1 + alpha2 - C);
      /* Equation 12.4 */
      H = std::min(C, alpha1 + alpha2);
    }

    if (L == H) return 0; 

    scalar_type k11 = base_type::kernel(points[i1], points[i1]);
    scalar_type k12 = base_type::kernel(points[i1], points[i2]);
    scalar_type k22 = base_type::kernel(points[i2], points[i2]);

    /* Equation 12.5 -- the second derivative of W, the objective function */
    double eta = 2 * k12 - k11 - k22;
    if (eta < 0) {
      /* Equation 12.6 */
      a2 = alpha2 - y2 * (e1-e2) / eta;
      /* Equation 12.7 */
      if (a2 < L) a2 = L;
      else if (a2 > H) a2 = H;
    }
    else {  /* This block handles corner cases; usually this means a training vector has been repeated */
      scalar_type f1 = operator()(points[i1]);
      scalar_type f2 = operator()(points[i2]);
      /* Equation 12.21 */
      scalar_type v1 = f1 + base_type::bias - target[i1] * alpha1 * k11 - target[i2] * alpha2 * k12;
      scalar_type v2 = f2 + base_type::bias - target[i1] * alpha1 * k12 - target[i2] * alpha2 * k22;
      /* Equation 12.22 */
      scalar_type gamma = alpha1 + s * alpha2;
      /* Equation 12.23 -- this is ugly and should maybe be refactored out? */
      scalar_type Lobj = gamma - s * L + L - .5 * k11 * (gamma - s * L) * (gamma - s * L) - .5 * k22 * L * L - s * k12 * (gamma - s * L) * L - target[i1] * (gamma - s * L) * v1 - target[i2] * L * v2;
      scalar_type Hobj = gamma - s * H + H - .5 * k11 * (gamma - s * H) * (gamma - s * H) - .5 * k22 * L * L - s * k12 * (gamma - s * H) * H - target[i1] * (gamma - s * H) * v1 - target[i2] * H * v2;
      
      /* Now we move the Lagrangian multipliers to the endpoint which has the highest value for W */
      if (Lobj > Hobj + EPS)
	a2 = L;
      else if (Lobj < Hobj - EPS)
	a2 = H;
      else
	a2 = alpha2;
    }

    /* If a2 is within epsilon of 0 or C, then call it a boundary example */
    if (a2 < .00000001)
      a2 = 0;
    else if (a2 > C - .00000001)
      a2 = C;    
    if (fabs(a2 - alpha2) < EPS*(a2 + alpha2 + EPS)) 
      return 0;
    

    /* Equation 12.8 */
    double a1 = alpha1 + s*(alpha2 - a2);
    if (a1 < 0) {
      a2 += s*a1;
      a1 = 0;
    }
    else if (a1 > C) {
      a2 += s * (a1 - C);
      a1 = C;
    }

    scalar_type old_bias = base_type::bias;
    /* Equation 12.9 */
    if (0 != a1 && C != a1)
      base_type::bias += e1 + y1 * (a1 - alpha1) * k11 + y2 * (a2 - alpha2) * k12;
    else {
      if (0 != a2 && C != a2)
	base_type::bias += e2 + y1 * (a1 - alpha1) * k12 + y2 * (a2 - alpha2) * k22;
      else 
	base_type::bias = ((base_type::bias + e1 + y1 * (a1 - alpha1) * k11 + y2 * (a2 - alpha2) * k12) +
			   (base_type::bias + e2 + y1 * (a1 - alpha1) * k12 + y2 * (a2 - alpha2) * k22)) / 2;
    }

    /* TODO figure out what to do about weight vectors for linear SVMs */

    /* Update the error cache */
    for (size_t i = 0; i < error_cache.size(); ++i) {
      if (0 != base_type::weight[i] && C != base_type::weight[i]) {
	error_cache[i] += y1 * (a1 - alpha1) * base_type::kernel(points[i1], points[i]) + target[i2] * (a2 - alpha2) * base_type::kernel(points[i2], points[i]) - (base_type::bias - old_bias);
      }
      else {
	/* This may not actually be necessary -- I'm not convinced any code paths will ever get here -- but since Platt
	   says we only keep cached error values if the Lagrange multiplier is non-bound, I'm covering my bases. */
	if (error_cache[i] > 0)
	  error_cache[i] = 0;
      }
    }
    /* p. 49, Platt: "When a Lagrange multiplier is non-bound and is involved in a joint optimization, its cached error
       is set to zero." But, hey, if it's a bound multiplier, we don't cache its error. So just make it zero anyway. */
    error_cache[i1] = 0;
    error_cache[i2] = 0;

    base_type::weight[i1] = a1;
    base_type::weight[i2] = a2;
    return 1;
  }
  
  int examineExample(int idx) {
    output_type y2 = target[idx];
    double alpha2 = base_type::weight[idx];
    result_type e2;
    if (alpha2 != 0 && alpha2 != C)
      e2 = error_cache[idx];
    else 
      e2 = operator()(points[idx]) - y2;
    
    result_type r2 = e2 * y2;
    if ((r2 < -tol && alpha2 < C) || (r2 > tol && alpha2 > 0)) {
      int count = std::count_if(base_type::weight.begin(), 
				base_type::weight.end(), 
				__gnu_cxx::compose2(std::logical_and<bool>(),
						    std::bind2nd(std::not_equal_to<scalar_type>(), 0),
						    std::bind2nd(std::not_equal_to<scalar_type>(), C)));

      if (count > 1) { // use second choice heuristic
	scalar_type tmp = 0, tmp2 = 0;
	int k = 0;
	for (size_t i = 0; i < error_cache.size(); ++i) {
	  tmp2 = fabs(error_cache[idx] - error_cache[i]);
	  if (tmp2 > tmp) {
	    tmp = tmp2;
	    k = i;
	  }
	}
	if (takeStep(idx, k)) 
	  return 1;	
      }

      for (size_t i = startpt(points.size()), j = i; i<j+points.size(); ++i) 
	if (base_type::weight[i%points.size()] != 0 && base_type::weight[i%points.size()] != C) 
	  if (takeStep(idx, i%points.size())) 
	    return 1;
	
      for (size_t i = startpt(points.size()), j = i; i<j+points.size(); ++i) 
	if (takeStep(idx, i%points.size())) 
	  return 1;
    }
    return 0;
  }
    
    template< class IRange, class ORange >
    void learn( IRange const &input, ORange const &output ) {
      points = input;
      target = output;
      base_type::weight.clear();
      base_type::weight.resize(points.size());
      base_type::support_vector.clear();
      base_type::support_vector.resize(points.size());
      error_cache.clear();
      error_cache.resize(points.size());
      
      int numChanged = 0;
      int examineAll = 1;
      while (numChanged > 0 || examineAll) {
	numChanged = 0;
	if (examineAll) 
	  for (size_t i=0; i < points.size(); ++i) 
	    numChanged += examineExample(i);
	else 
	  for (size_t i=0; i < points.size(); ++i)
	    if (base_type::weight[i] != 0 && base_type::weight[i] != C) 
	      numChanged += examineExample(i);

	if (1 == examineAll) 
	  examineAll = 0;
	else if (0 == numChanged) 
	  examineAll = 1;
      }
    }

    void printweights() {
      for (int i=0; i<base_type::weight.size(); ++i)
	std::cout << base_type::weight[i] << " ";
      std::cout << std::endl;
    }
    
    scalar_type C;
    scalar_type tol;
    std::vector<double> error_cache;
    std::vector<input_type> points;
    std::vector<output_type> target;
    boost::mt19937 randomness;
    boost::random_number_generator<boost::mt19937> startpt;
  };

  // Ranking SVM. TODO: Test ASAP!
  /*
template<typename I, typename O, template<typename,int> class K>
class svm<I,O,K, typename boost::enable_if<boost::is_same<O,int> >::type>:
    public determinate<I,O,K> {
  */

  template<typename Problem, template<typename, int> class K>
  class svm<Problem, K, typename boost::enable_if<is_ranking<Problem> >::type >:
    public determinate<typename Problem::input_type, typename Problem::output_type, K> {
  public:
    typedef determinate<typename Problem::input_type, typename Problem::output_type, K> base_type;
    typedef typename base_type::kernel_type kernel_type;
    typedef double scalar_type;
    typedef typename Problem::input_type input_type;
    typedef typename Problem::output_type output_type;  
    typedef typename base_type::result_type result_type;

    svm( typename boost::call_traits<kernel_type>::param_type k,
	 typename boost::call_traits<double>::param_type max_weight ): 
      base_type(k), C(max_weight), inner_machine(k, max_weight) {}

    result_type operator()(input_type const &x) {
      result_type ret;
      for (int i=0; i<base_type::weight.size(); ++i)
	if (base_type::weight[i] > 0)
	  ret += base_type::weight[i] * target[i] * base_type::kernel(points[i], x);
      ret += base_type::bias;
      return ret;
    }

    template<class IRange, class ORange>
    void learn(IRange const &input, ORange const &output) {
      std::vector<input_type> points;
      std::vector<bool> target;
      for (int i = input.begin(); i < input.size(); ++i)
	for (int j = i+1; j < input.size(); ++j)
	  if (output[i] != output[j]) {
	    input_type diff_vec;
	    std::transform(input[i].begin(), input[i].end(), input[j].begin, diff_vec.begin(),
			   std::minus<typename boost::range_value<input_type>::type>());
	    points.push_back(diff_vec);
	    target.push_back(output[i] > output[j]);
	  }
      inner_machine.learn(points, target);
      base_type::weight = inner_machine.weight;
      base_type::support_vector = inner_machine.support_vector;
      base_type::bias = inner_machine.bias;
    }

    scalar_type epsilon;
    scalar_type C;
    std::vector<input_type> points;
    std::vector<output_type> target;

    typedef kml::classification<input_type, bool> problem_type;
    svm<problem_type, K> inner_machine;
  };

} // namespace kml
#endif
