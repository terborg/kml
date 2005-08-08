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

#include <boost/numeric/bindings/traits/std_vector.hpp>
#include <boost/numeric/bindings/traits/ublas_vector.hpp>
#include <boost/mpl/equal_to.hpp>
#include <boost/type_traits/is_float.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/random_number_generator.hpp>

#include <algorithm>
#include <vector>

#include <kml/determinate.hpp>



namespace kml {

  template<typename I, typename O, template<typename, int> class K, class Enable = void >
  class svm: public determinate<I,O,K> {};

  // Classification SVM

template< typename I, typename O, template<typename,int> class K >
class svm<I,O,K, typename boost::enable_if<boost::is_same<O, bool> >::type>: 
    public determinate<I,O,K> {
public:
  typedef determinate<I,O,K> base_type;
  typedef typename base_type::kernel_type kernel_type;
  typedef typename base_type::result_type result_type;
  typedef I input_type;
  typedef O output_type;  
  typedef double scalar_type;


  // Bug fixed: use call_traits instead of const&, so can be called with both reference and non-reference
  svm( typename boost::call_traits<kernel_type>::param_type k,
       typename boost::call_traits<double>::param_type max_weight ): 
    base_type(k), C(max_weight), startpt(randomness) {}

  result_type operator() (input_type const &x) {
    // coming soon
    return 0;
  }

  int takeStep(int i1, int i2) {
    if (i1 == i2) return 0;

    double alpha1 = base_type::weight[i1];
    output_type y1 = target[i1];
    /* p. 49, Platt: "When an error E is required by SMO, it will look up the error in the error cache if the 
       corresponding Lagrange multiplier is not at bound." */
    if (0 != base_type::weights[i1] && C != base_type::weights[i1]) 
      scalar_type e1 = error_cache[i1];
    else
      scalar_type e1 = operator()(points[i1]) - y1;

    double alpha2 = base_type::weight[i2];
    output_type y2 = target[i2];
    /* p. 49 again */
    if (0 != base_type::weights[i2] && C != base_type::weights[i2])
      scalar_type e2 = error_cache[i2];
    else
      scalar_type e2 = (double)operator()(points[i1]) - y2;
    
    output_type s = y1 * y2;

    if (y1 != y2) {
      scalar_type L = max(0, alpha2 - alpha1);
      scalar_type H = min(C, C + alpha2 - alpha1);
    }
    else {
      /* Equation 12.3 */
      double L = max(0, alpha1 + alpha2 - C);
      /* Equation 12.4 */
      double H = min(C, alpha1 + alpha1);
    }

    if (L == H) return 0;

    output_type k11 = base_type::kernel(points[i1], points[i1]);
    output_type k12 = base_type::kernel(points[i1], points[i2]);
    output_type k22 = base_type::kernel(points[i2], points[i2]);
    /* Equation 12.5 -- the second derivative of W, the objective function */
    double eta = 2 * k12 - k11 - k22;

    if (eta < 0) {
      /* Equation 12.6 */
      double a2 = alpha2 - y2 * (e1-e2) / eta;
      /* Equation 12.7 */
      if (a2 < L) a2 = L;
      else if (a2 > H) a2 = H;
    }
    else {  /* This block handles corner cases; usually this means a training vector has been repeated */
      output_type f1 = operator()(points[i1]);
      output_type f2 = operator()(points[i2]);
      /* Equation 12.21 */
      scalar_type v1 = f1 + base_type::bias - target[i1] * alpha1 * k11 - target[i2] * alpha2 * k12;
      scalar_type v2 = f2 + base_type::bias - target[i1] * alpha1 * k12 - target[i2] * alpha2 * k22;
      /* Equation 12.22 */
      scalar_type gamma = alpha1 + s * alpha2;
      /* Equation 12.23 -- this is ugly and should maybe be refactored out? */
      scalar_type Lobj = gamma - s * L + L - .5 * k11 * (gamma - s * L) * (gamma - s * L) - .5 * k22 * L * L - s * k12 * (gamma - s * L) * L - target[i1] * (gamma - s * L) * v1 - target[i2] * L * v2;
      scalar_type Hobj = gamma - s * H + H - .5 * k11 * (gamma - s * H) * (gamma - s * H) - .5 * k22 * L * L - s * k12 * (gamma - s * H) * H - target[i1] * (gamma - s * H) * v1 - target[i2] * H * v2;
      
      /* Now we move the Lagrangian multipliers to the endpoint which has the highest value for W */
      if (Lobj > Hobj + eps)
	double a2 = L;
      else if (Lobj < Hobj - eps)
	double a2 = H;
      else
	double a2 = alpha2;
    }

    /* If a2 is within epsilon of 0 or C, then call it a boundary example */
    if (a2 < .00000001)
      a2 = 0;
    else if (a2 > C - .00000001)
      a2 = C;
    if (abs(a2 - alpha2) < eps*(a2 + alpha2 + eps))
      return 0;

    /* Equation 12.8 */
    double a1 = alpha1 + s*(alpha2 - a2);

    /* TODO Update threshold to reflect change in Lagrange multipliers */
    scalar_type old_bias = base_type::bias;
    /* Equation 12.9 */
    if (0 != a1 && C != a1)
      base_type::bias += E1 + target[i1] * (a1 - alpha1) * k11 + target[i2] * (a2 - alpha2) * k12;
    else {
      if (0 != a2 && C != a2)
	base_type::bias += E2 + target[i1] * (a1 - alpha1) * k12 + target[i2] * (a2 - alpha2) * k22;
      else
	base_type::bias = ((base_type::bias + E1 + target[i1] * (a1 - alpha1) * k11 + target[i2] * (a2 - alpha2) * k12) +
			   (base_type::bias + E2 + target[i1] * (a1 - alpha1) * k12 + target[i2] * (a2 - alpha2) * k22)) / 2;
    }
    /* TODO figure out what to do about weight vectors for linear SVMs */

    /* Update the error cache */
    for (int i = 0; i < error_cache.size(); ++i) {
      if (0 != base_type::weight[i] && C != base_type::weight[i])
	error_cache[i] += target[i1] * (a1 - alpha1) * base_type::kernel(points[i1], points[i]) + target[i2] * (a2 - alpha2) * base_type::kernel(points[i2], points[i]) + old_bias - base_type::bias;
      else
	/* This may not actually be necessary -- I'm not convinced any code paths will ever get here -- but since Platt
	   says we only keep cached error values if the Lagrange multiplier is non-bound, I'm covering my bases. */
	error_cache[i] = 0;
    }
    /* p. 49, Platt: "When a Lagrange multiplier is non-bound and is involved in a joint optimization, its cached error
       is set to zero." But, hey, if it's a bound multiplier, we don't cache its error. So just make it zero anyway. */
    error_cache[i1] = 0;
    error_cache[i2] = 0;

    base_type::weight[i1] = a1;
    base_type::weight[i2] = a2;
  }
  
  int examineExample(int idx) {
    output_type y2 = target[idx];
    double alpha2 = base_type::weight[idx];
    if (alpha2 != 0 && alpha1 != C)
      result_type e2 = error_cache[idx];
    else
      result_type e2 = operator()(points[idx]) - y2;

    result_type r2 = e2 * y2;
    if ((r2 < -(base_type::bias) && alpha2 < C) || 
	(r2 > base_type::bias && alpha2 > 0)) {
      int count = std::count_if(points.begin(), points.end(), std::bind2nd(std::equal_to<scalar_type>(), 0));
      if (count <= 1)
	count += std::count_if(points.begin(), points.end(), std::bind2nd(std::equal_to<scalar_type>(), C));
      if (count > 1) {
	int i = 0; // TODO second choice heuristic stuff
	if (takeStep(i, idx))
	  return 1;
      }
      for (int i = startpt(points.size()),
	     j = i; i % points.size() != j; ++i)
	if (base_type::weight[i] != 0 && base_type::weight[i] != C)
	  if (takeStep(i, idx))
	    return 1;
      for (int i = startpt(points.size()),
	     j = i; i % points.size() != j; ++i)
	if (takeStep(i, idx))
	  return 1;
    }
    return 0;
  }

  template< class IRange, class ORange >
  void learn( IRange const &input, ORange const &output ) {
    points = input;
    target = output;
    base_type::weight = 0.0;
    base_type::support_vector.resize(points.size());

    int numChanged = 0;
    int examineAll = 1;
    while (numChanged > 0 || examineAll) {
      numChanged = 0;
      if (examineAll)
	for (int i=0; i < points.size(); ++i)
	  numChanged += examineExample(i);
      else
	for (int i=0; i < points.size(); ++i)
	  if (base_type::support_vector[i] != 0 && base_type::support_vector[i] != C)
	    numChanged += examineExample(i);
      if (1 == examineAll)
	examineAll = 0;
      else if (0 == numChanged)
	examineAll = 1;
    }
  }

  scalar_type C;
  std::vector<double> error_cache;
  std::vector<input_type> points;
  std::vector<output_type> target;
  boost::mt19937 randomness;
  //  boost::normal_distribution<int> norm_dist;
  //  boost::variate_generator<boost::mt19937, boost::normal_distribution<int> > startpt;
  boost::random_number_generator<boost::mt19937> startpt;
};

  // Ranking SVM

template<typename I, typename O, template<typename,int> class K>
class svm<I,O,K, typename boost::enable_if<boost::is_same<O,int> >::type>:
    public determinate<I,O,K> {
public:
  typedef determinate<I,O,K> base_type;
  typedef typename base_type::kernel_type kernel_type;
  typedef double scalar_type;

  svm( typename boost::call_traits<kernel_type>::param_type k,
       typename boost::call_traits<double>::param_type max_weight ): 
    base_type(k), C(max_weight), inner_machine(k, max_weight) {}

  template<class IRange, class ORange>
  void learn(IRange const &input, ORange const &output) {

  }

  scalar_type epsilon;
  scalar_type C;
  svm<I,bool,K> inner_machine;
};

} // namespace kml
#endif
