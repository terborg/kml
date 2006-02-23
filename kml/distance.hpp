/***************************************************************************
 *  The Kernel-Machine Library                                             *
 *  Copyright (C) 2004--2006 by Rutger W. ter Borg                         *
 *                                                                         *
 *  This library is free software; you can redistribute it and/or          *
 *  modify it under the terms of the GNU Lesser General Public             *
 *  License as published by the Free Software Foundation; either           *
 *  version 2.1 of the License, or (at your option) any later version.     *
 *                                                                         *
 *  This library is distributed in the hope that it will be useful,        *
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of         *
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU      *
 *  Lesser General Public License for more details.                        *
 *                                                                         *
 *  You should have received a copy of the GNU Lesser General Public       *
 *  License along with this library; if not, write to the Free Software    *
 *  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  *
 ***************************************************************************/

#ifndef DISTANCE_HPP
#define DISTANCE_HPP


/*!
 
-------> Old code, should be revised

Basic functionality to compute the distance of scalar types and vector types. 
It automatically chooses an efficient implementation,
on the basis of the HAVE_BOOST_NUMERIC_BINDINGS define, and the data
type passed by the user.
 
Available algorithms:
 
- \b distance. Computes ||u-v||_2
- \b distance_square. Computes ||u-v||_2^2
 

Template metaprogramming is used to make one generic function work on both scalar and
vector types.

This should work on any kernel function.

 
*/


#include <boost/lambda/bind.hpp>
#include <boost/mpl/eval_if.hpp>
#include <boost/mpl/if.hpp>
#include <boost/mpl/identity.hpp>
#include <boost/range/begin.hpp>
#include <boost/range/end.hpp>
#include <boost/range/size_type.hpp>
#include <boost/range/size.hpp>
#include <boost/range/value_type.hpp>
#include <boost/type_traits/is_scalar.hpp>

#include <kml/input_value.hpp>

#include <cmath>
#include <functional>
#include <numeric>

namespace mpl = boost::mpl;
namespace lambda = boost::lambda;

// TODO use call_traits for best possible parameter and return type selection
// however, compiler seems to complain (at a first try)


namespace kml { 
  namespace detail {


    template<typename Range>
    struct range_comp: public std::binary_function<Range,Range,bool> {
      bool operator()(Range const &u, Range const &v) {
	typename Range::const_iterator iter_u = u.begin();
	typename Range::const_iterator iter_v = v.begin();
	while( (iter_u != u.end()) && (*iter_u == *iter_v) ) {
	  ++iter_u;
	  ++iter_v;
	}
	if (iter_u == u.end())
	  return false;
	else
	  return *iter_u < *iter_v;
      }
    };
    
    template<typename Range>
    struct range_equal: public std::binary_function<Range,Range,bool> {
      bool operator()(Range const &u, Range const &v) {
	return ( distance_square( u,v) < 1e-6 );
	
      }
    };
    
    
    
    template<typename Range>
    struct has_NaNs: public std::unary_function<Range,bool> {
      bool operator()(Range const &u) {
	typedef typename boost::range_value<Range>::type scalar_type;
	return (std::find_if( boost::begin(u), boost::end(u), lambda::bind( std::not_equal_to<scalar_type>(),lambda::_1,lambda::_1) ) != u.end());
      }
    };
    
    
    struct scalar_square {
      template <typename T>
      inline
      T operator()(const T& x) {
	return x * x;
      }
    };
    
    struct scalar_distance_square {
      template <typename T>
      static T compute(const T& x, const T& y) {
        static scalar_square ss;
	return ss(x-y);
      }
    };
    
    struct vector_distance_square {
      template<typename T>
      inline
      static typename boost::range_value<T>::type compute( T const &u, T const &v ) {
        static scalar_square ss;
	typename boost::range_value<T>::type result(0);
	for( typename boost::range_size<T>::type i = 0; i < boost::size(u); ++i ) {
	  result += ss(u[i]-v[i]);
	}
	return result;
      }
    };
    
  } // namespace detail
  
  
	/*! distance_square function which works on both scalars and vectors.
		Complex numbers not taken into account :-)
		\ingroup helper
	*/
	template<typename T>
	typename input_value<T>::type
	distance_square( T const &u, T const &v ) {
		return mpl::if_<
		boost::is_scalar<T>,
		detail::scalar_distance_square,
		detail::vector_distance_square
		>::type::compute( u, v );
	}
  

  template<typename T,typename Kernel>
  struct closer_by: std::binary_function< T, T, bool > {
  	closer_by(T const &c, Kernel &k): location(c), kernel(k) {}
  	bool operator()( T const &a, T const &b ) {
		return (kernel(a,a)-2.0*kernel(location,a)) < (kernel(b,b)-2.0*kernel(location,b));
	}
  	T location;
	Kernel kernel;
  };
 

} // namespace kml


#endif
