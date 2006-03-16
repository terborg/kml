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
#define sgn(a)     (((a) < 0) ? -1 : ((a) > 0) ? 1 : 0)

#include <boost/numeric/bindings/traits/std_vector.hpp>
#include <boost/numeric/bindings/traits/ublas_vector.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/random_number_generator.hpp>
#include <boost/range/value_type.hpp>
#include <boost/lambda/lambda.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/vector_property_map.hpp>
#include <boost/numeric/ublas/vector.hpp>

#include <algorithm>
#include <vector>
#include <iostream>
#include <utility>
#include <iterator>

#include <kml/kernel_machine.hpp>
#include <kml/classification.hpp>
#include <kml/ranking.hpp>
#include <kml/regression.hpp>

namespace lambda = boost::lambda;
namespace ublas = boost::numeric::ublas;

namespace kml {

  template<typename Problem, typename Kernel, typename PropertyMap, class Enable = void>
  class svm: public kernel_machine<Problem, Kernel, PropertyMap> {};
  
  // Classification SVM
  
  template<typename Problem, typename Kernel, typename PropertyMap>
  class svm<Problem, Kernel, PropertyMap, typename boost::enable_if<is_classification<Problem> >::type>:
    public kernel_machine<Problem, Kernel, PropertyMap> {
  public:
    typedef kernel_machine<Problem, Kernel, PropertyMap> base_type;
    typedef typename base_type::kernel_type kernel_type;
    typedef typename base_type::result_type result_type;
    typedef typename Problem::input_type input_type; 
    typedef typename Problem::output_type output_type;  
    typedef double scalar_type;
    
    /*
    typedef typename boost::property_traits<PropertyMap>::key_type key_type;
    typedef typename boost::property_traits<PropertyMap>::value_type value_type;
    */
    svm( typename boost::call_traits<kernel_type>::param_type k,
	 typename boost::call_traits<scalar_type>::param_type max_weight,
	 typename boost::call_traits<PropertyMap>::param_type map ): 
      base_type(k, map), C(max_weight), tol(.001), startpt(randomness) { }
    
    result_type operator() (input_type const &x) {
      scalar_type ret=0;
      for (size_t i=0; i<weight.size(); ++i)
	if (weight[i] > 0)
	  ret += weight[i] * (*base_type::data)[i].get<1>() * base_type::kernel(i, x);
      ret -= bias;
      return sgn(ret);
    }
    
    int takeStep(int i1, int i2) {
      if (i1 == i2) return 0; 
      
      double alpha1 = weight[i1];
      output_type y1 = (*base_type::data)[i1].get<1>();
      scalar_type e1, e2, L, H, a2;
      /* p. 49, Platt: "When an error E is required by SMO, it will look up the error in the error cache if the 
	 corresponding Lagrange multiplier is not at bound." */
      if (0 != alpha1 && C != alpha1) 
	e1 = error_cache[i1];
      else
	e1 = operator()((*base_type::data)[i1].get<0>()) - y1;
      
      double alpha2 = weight[i2];
      output_type y2 = (*base_type::data)[i2].get<1>();
      /* p. 49 again */
      if (0 != alpha2 && C != alpha2)
	e2 = error_cache[i2];
      else
	e2 = (double)operator()((*base_type::data)[i2].get<0>()) - y2;
      
      scalar_type s = y1 * y2;
      
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
      
      scalar_type k11 = base_type::kernel(i1, i1);
      scalar_type k12 = base_type::kernel(i1, i2);
      scalar_type k22 = base_type::kernel(i2, i2);
      
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
	scalar_type f1 = operator()((*base_type::data)[i1].get<0>());
	scalar_type f2 = operator()((*base_type::data)[i2].get<0>());
	/* Equation 12.21 */
	scalar_type v1 = f1 + bias - (*base_type::data)[i1].get<1>() * alpha1 * k11 - (*base_type::data)[i2].get<1>() * alpha2 * k12;
	scalar_type v2 = f2 + bias - (*base_type::data)[i1].get<1>() * alpha1 * k12 - (*base_type::data)[i2].get<1>() * alpha2 * k22;
	/* Equation 12.22 */
	scalar_type gamma = alpha1 + s * alpha2;
	/* Equation 12.23 -- this is ugly and should maybe be refactored out? */
	scalar_type Lobj = gamma - s * L + L - .5 * k11 * (gamma - s * L) * (gamma - s * L) - .5 * k22 * L * L - s * k12 * (gamma - s * L) * L - (*base_type::data)[i1].get<1>() * (gamma - s * L) * v1 - (*base_type::data)[i2].get<1>() * L * v2;
	scalar_type Hobj = gamma - s * H + H - .5 * k11 * (gamma - s * H) * (gamma - s * H) - .5 * k22 * L * L - s * k12 * (gamma - s * H) * H - (*base_type::data)[i1].get<1>() * (gamma - s * H) * v1 - (*base_type::data)[i2].get<1>() * H * v2;
	
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
      
      scalar_type old_bias = bias;
      /* Equation 12.9 */
      if (0 != a1 && C != a1)
	bias += e1 + y1 * (a1 - alpha1) * k11 + y2 * (a2 - alpha2) * k12;
      else {
	if (0 != a2 && C != a2)
	  bias += e2 + y1 * (a1 - alpha1) * k12 + y2 * (a2 - alpha2) * k22;
	else 
	  bias = ((bias + e1 + y1 * (a1 - alpha1) * k11 + y2 * (a2 - alpha2) * k12) +
		  (bias + e2 + y1 * (a1 - alpha1) * k12 + y2 * (a2 - alpha2) * k22)) / 2;
      }
      
      /* TODO figure out what to do about weight vectors for linear SVMs */
      
      /* Update the error cache */
      for (size_t i = 0; i < error_cache.size(); ++i) {
	if (0 != weight[i] && C != weight[i]) {
	  error_cache[i] += y1 * (a1 - alpha1) * base_type::kernel(i1, i) + (*base_type::data)[i2].get<1>() * (a2 - alpha2) * base_type::kernel(i2, i) - (bias - old_bias);
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
      
      weight[i1] = a1;
      weight[i2] = a2;
      return 1;
    }
    
    int examineExample(int idx) {
      output_type y2 = (*base_type::data)[idx].get<1>();
      double alpha2 = weight[idx];
      scalar_type e2;
      if (alpha2 != 0 && alpha2 != C)
	e2 = error_cache[idx];
      else 
	e2 = operator()((*base_type::data)[idx].get<0>()) - y2;
      
      scalar_type r2 = e2 * y2;
      if ((r2 < -tol && alpha2 < C) || (r2 > tol && alpha2 > 0)) {
	int count = std::count_if(weight.begin(), weight.end(),
				  (lambda::_1 != 0) && (lambda::_1 != C) );
	
	
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

	for (size_t i = startpt(size), j = i; i<j+size; ++i) 
	  if (weight[i%size] != 0 && weight[i%size] != C) 
	    if (takeStep(idx, i%size)) 
	      return 1;
	
	for (size_t i = startpt(size), j = i; i<j+size; ++i) 
	  if (takeStep(idx, i%size)) 
	    return 1;
      }
      return 0;
    }
    
    template<typename KeyIterator>
    void learn(KeyIterator begin, KeyIterator end) {
      size = std::distance(begin, end);
      weight.clear();
      weight.resize(size);
      error_cache.clear();
      error_cache.resize(size);
     
      int numChanged = 0;
      int examineAll = 1;
      while (numChanged > 0 || examineAll) {
	numChanged = 0;
	if (examineAll) 
	  for (size_t i=0; i < size; ++i) 
	    numChanged += examineExample(i);
	else 
	  for (size_t i=0; i < size; ++i)
	    if (weight[i] != 0 && weight[i] != C) 
	      numChanged += examineExample(i);

	if (1 == examineAll) 
	  examineAll = 0;
	else if (0 == numChanged) 
	  examineAll = 1;
      }
    }

    void printweights() {
      for ( size_t i=0; i<weight.size(); ++i)
	std::cout << weight[i] << " ";
      std::cout << std::endl;
    }
    
    unsigned int size;
    scalar_type C;
    std::vector<scalar_type> weight;
    scalar_type tol;
    scalar_type bias;
    std::vector<scalar_type> error_cache;
    boost::mt19937 randomness;
    boost::random_number_generator<boost::mt19937> startpt;
  };

  // Ranking SVM. 

  template<typename Problem, typename Kernel, typename PropertyMap>
  class svm<Problem, Kernel, PropertyMap, typename boost::enable_if<is_ranking<Problem> >::type >:
    public kernel_machine<Problem, Kernel, PropertyMap> {
  public:
    typedef kernel_machine<Problem, Kernel, PropertyMap> base_type;
    typedef typename base_type::kernel_type kernel_type;
    typedef double scalar_type;
    typedef typename Problem::input_type input_type;
    typedef typename Problem::output_type output_type;  
    typedef typename base_type::result_type result_type;

    /*
    typedef typename boost::property_traits<PropertyMap>::key_type key_type;
    typedef typename boost::property_traits<PropertyMap>::value_type value_type;
    */

    svm( typename boost::call_traits<kernel_type>::param_type k,
	 typename boost::call_traits<double>::param_type max_weight,
	 typename boost::call_traits<PropertyMap>::param_type map): 
      base_type(k, map), C(max_weight), inner_machine(k, max_weight, map) {}

    /* hm, this is interesting: we'll have to remember to reset the data to the new propertymap */

    result_type operator()(input_type const &x) {
      return inner_machine(x);
    }


    template<typename KeyIterator>
    void learn(KeyIterator begin, KeyIterator end) {
      size = std::distance(begin, end);
      std::vector<input_type> points;
      std::vector<int> target;
      for (unsigned int i = 0; i < size-1; ++i)  // no need to compare against the last point, we already did
	for (unsigned int j = i+1; j < size; ++j)
	  if ((*base_type::data)[i].get<1>().first == (*base_type::data)[j].get<1>().first) 
	    if ((*base_type::data)[i].get<1>().second != (*base_type::data)[j].get<1>().second) {
	      input_type diff_vec;
	      std::transform((*base_type::data)[i].get<0>().begin(), 
			     (*base_type::data)[i].get<0>.end(), 
			     (*base_type::data)[j].get<0>().begin(), 
			     std::back_inserter(diff_vec),
			     std::minus<typename boost::range_value<input_type>::type>());
	      /* problem: we actually need to be populating a new propertymap */
	      points.push_back(diff_vec);
	      target.push_back(sgn((*base_type::data)[i].get<1>().second - (*base_type::data)[j].get<1>().second));
	    }

      /* Let's see if artificially creating a second class works */

      bool one_class = true;
      for (unsigned int i=1; i<target.size(); ++i) 
	if (target[i] != target[0]) {
	  one_class = false;
	  break;
	}

      if (one_class) 
	for (unsigned int i=0; i<target.size(); i=i+2) {
	  target[i] = -target[i];
	  for (unsigned int j=0; j<points[i].size(); ++j)
	    points[i][j] = -points[i][j];
	}

      /* done */

      inner_machine.learn(points, target);
    }

    unsigned int size;
    scalar_type epsilon;
    scalar_type C;
    std::vector<input_type> points;
    std::vector<output_type> target;

    typedef boost::tuple<ublas::vector<input_type>, int> example_type;
    typedef kml::classification<example_type> problem_type;
    svm<problem_type, Kernel, PropertyMap> inner_machine;
  };

} // namespace kml
#endif
