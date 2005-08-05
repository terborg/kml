/***************************************************************************
 *  The Kernel-Machine Library                                             *
 *  Copyright (C) 2004, 2005 by Rutger W. ter Borg                         *
 *                                                                         *
 *  This program is free software; you can redistribute it and/or          *
 *  modify it under the terms of the GNU General Public License            *
 *  as published by the Free Software Foundation; either version 2         *
 *  of the License, or (at your option) any later version.                 *
 *                                                                         *
 *  This program is distributed in the hope that it will be useful,        *
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of         *
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the          *
 *  GNU General Public License for more details.                           *
 *                                                                         *
 *  You should have received a copy of the GNU General Public License      *
 *  along with this program; if not, write to the Free Software            *
 *  Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307  *
 ***************************************************************************/

#ifndef SVM_HPP
#define SVM_HPP

#include <traits.hpp>
#include <algorithm/aosvr.hpp>
#include <determinate.hpp>



namespace kml {

  // Classification SVM

template<typename I, typename O, template<typename,int> class K >
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
    base_type(k), C(max_weight) {}

  int takeStep(int i1, int i2) {
    if (i1 == i2) return 0;

    double alpha1 = base_type::weight[i1];
    output_type y1 = target[i1];
    if (alpha1 > 0 && alpha1 < C) // This is what Hwanjo did; original code
      result_type e1 = error_cache[i1]; // has a different test
    else
      result_type e1 = operator()(points[i1]) - y1;

    double alpha2 = base_type::weight[i2];
    output_type y2 = target[i2];
    if (alpha2 > 0 && alpha2 < C)
      result_type e2 = error_cache[i2];
    else
      result_type e2 = operator()(points[i1]) - y2;
    
    output_type s = y1 * y2;

    if (y1 != y2) {
      scalar_type L = max(0, alpha2 - alpha1);
      scalar_type H = min(C, C + alpha2 - alpha1);
    }
    else {
      double L = max(0, alpha1 + alpha2 - C);
      double H = min(C, alpha1 + alpha1);
    }

    if (L == H) return 0;

    output_type k11 = base_type::kernel(points[i1], points[i1]);
    output_type k12 = base_type::kernel(points[i1], points[i2]);
    output_type k22 = base_type::kernel(points[i2], points[i2]);
    output_type eta = 2*k12-k11-k22;

    if (eta < 0) {
      double a2 = alpha2 - y2 * (e1-e2) / eta; // Hwanjo's code is different...
      if (a2 < L) a2 = L;
      else if (a2 > H) a2 = H;
    }
    else {
      scalar_type c1 = (double)eta / 2;
      scalar_type c2 = y2 * (e1 - e2) - eta * alpha2;
      scalar_type Lobj = c1 * L * L + c2 * L;
      scalar_type Hobj = c1 * H * H + c2 * H;
      
      if (Lobj > Hobj + eps)
	double a2 = L;
      else if (Lobj < Hobj - eps)
	double a2 = H;
      else
	double a2 = alpha2;
    }

    if (a2 < .00000001)
      a2 = 0;
    else if (a2 > C - .00000001)
      a2 = C;
    if (abs(a2 - alpha2) < eps*(a2 + alpha2 + eps))
      return 0;

    double a1 = alpha1 + s*(alpha2 - a2);

    /* TODO Update threshold (base_type::bias?) to reflect change in 
       Lagrange multipliers */
    /* TODO figure out what to do about weight vectors for linear SVMs */
    /* TODO update error cache, now that we have one ;) */

    base_type::weight[i1] = a1;
    base_type::weight[i2] = a2;
  }  

  int examineExample(int idx) {
    output_type y2 = target[idx];
    double alpha2 = base_type::weight[idx];
    if (alpha2 > 0 && alpha1 < C)
      result_type e2 = error_cache[idx];
    else
      result_type e2 = operator()(points[idx]) - y2;

    result_type r2 = e2 * y2;
    if ((r2 < -(base_type::bias) && alpha2 < C) || 
	(r2 > base_type::bias && alpha2 > 0)) {
      typename vector<input_type>::iterator it = std::find_if(points.begin(),
							      points.end(),
							      /* TODO predicate */);
      if (it != points.end() && std::find_if(it, points.end(), /* TODO predicate */) != points.end()) {
	int i = 0; // TODO second choice heuristic stuff
	if (takeStep(i, idx))
	  return 1;
      }
      for (int i = /* TODO see boost::random - between 0 and sv.size() */,
	     int j = i; i % points.size() != j; ++i)
	if (base_type::weight[i] != 0 && base_type::weight[i] != C)
	  if (takeStep(i, idx))
	    return 1;
      for (int i = /* TODO see boost::random */,
	     int j = i; i % points.size() != j; ++i)
	if (takeStep(i, idx))
	  return 1;
    }
    return 0;
  }

  template< class IRange, class ORange >
  void learn( IRange const &input, ORange const &output ) {
    points = input;
    target = output;

    /* and a whole lotta other stuff */
    
  }

  scalar_type C;
  vector<result_type> error_cache;
  vector<input_type> points;
  vector<output_type> target;
};

  // Ranking SVM

template<typename I, typename O, template<typename,int> class K>
class svm<I,O,K, typename boost::enable_if<boost::is_same<O, int> >::type>:
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
}

#endif
