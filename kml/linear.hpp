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

#ifndef LINEAR_HPP
#define LINEAR_HPP

#include <boost/mpl/arithmetic.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/mpl/divides.hpp>
#include <boost/mpl/eval_if.hpp>
#include <boost/mpl/identity.hpp>
#include <boost/mpl/if.hpp>
#include <boost/mpl/math/is_even.hpp>
#include <boost/numeric/bindings/atlas/cblas.hpp>
#include <boost/range/begin.hpp>
#include <boost/range/end.hpp>
#include <boost/range/value_type.hpp>
#include <boost/range/size.hpp>
#include <boost/type_traits/is_scalar.hpp>
#include <boost/serialization/tracking.hpp>
#include <boost/serialization/access.hpp>

#include <kml/power_value.hpp>
#include <numeric>

namespace atlas = boost::numeric::bindings::atlas;
namespace mpl = boost::mpl;

/*!
Linear kernel
 
Template metaprogramming is used to make one generic function work on both scalar and
vector types.
 
\todo 
- ensure the atlas-specific implementation is not deployed on non-supported types
 
*/



namespace kml {
namespace detail {

class scalar_inner_prod {
public:
    template<typename T>
    static T compute( T const &x, T const &y ) {
        return (x * y);
    }
};


class vector_inner_prod {
public:
    template<typename T>
    static typename boost::range_value<T>::type compute( T const &u, T const &v ) {
        return atlas::dot( u, v );
    }
};





template<int N>
class power_even {
public:
    template<typename T>
    static typename boost::range_value<T>::type compute( T const &x ) {
        return std::pow( inner_product(x,x), N>>1 );
    }
};



template<int N>
class power_uneven {
public:
    template<typename T>
    static T compute( T const &x ) {
        return power_even<N>::compute(x) * x;
    }
};












} // namespace detail





/*!
\brief Linear kernel
\param T defines the argument type
 
The linear kernel is the basic Eucledian inner product.
 
*/


template<typename T>
struct linear: public std::binary_function<T,T,typename input_value<T>::type> {

    /*! Refinement of AdaptableBinaryFunction */
    typedef T first_argument_type;
    typedef T second_argument_type;
    typedef typename input_value<T>::type result_type;

    /*! Refinement of Kernel ? */
    typedef linear<T> type;
    typedef typename input_value<T>::type scalar_type;
    friend class boost::serialization::access;

    /*! Refinement of DefaultConstructible */
    linear() {}

    /*! Refinement of CopyConstructible */
    linear( type const &other ) {}

    /*! Refinement of Assignable */
    type &operator=( type const &other ) {
        return *this;
    }

    /*! Kernel constructor by providing TokenIterators */
    template<typename TokenIterator>
    linear( TokenIterator const begin, TokenIterator const end ) {}

    /*! Returns the result of the evaluation of a linear kernel on two input patters of type T
        \param u input pattern u
        \param v input pattern v
        \return u^T v               */
    inline
    scalar_type operator()( T const &u, T const &v ) const {
        return mpl::if_<boost::is_scalar<T>,
               detail::scalar_inner_prod,
               detail::vector_inner_prod >::type::compute( u, v );
    }

    // loading and saving capabilities
    // basically a dummy; we don't need to load or save anything
    template<class Archive>
    void serialize( Archive &archive, unsigned int const version ) {}

    // for debugging purposes
    friend std::ostream& operator<<(std::ostream &os, type const &) {
        os << "Linear kernel (u^T * v)" << std::endl;
        return os;
    }

};




template<class T, int N>
struct power_return_type: mpl::eval_if<
                               mpl::is_even< mpl::int_<N> >,
                               boost::range_value<T>,
                       mpl::identity<T> > {}
                   ;


template<typename Vector, int N>
typename power_return_type<Vector,N>::type power( Vector const &x ) {
                         return mpl::if_<
                                mpl::is_even< mpl::int_<N> >,
                                detail::power_even<N>,
                                detail::power_uneven<N> >::type::compute( x );
                     }


  /* Define templates is_linear so we can recognise the linear kernel in use
     when we need to, e.g. in ranking SVMs */

  /* I think this is not the right way to go about it; read Josuttis at home */
  template<typename T>
  struct is_linear : boost::mpl::bool_<false> {};

  template<typename T>
  struct is_linear<linear<T> > : boost::mpl::bool_<true> {};


} // namespace kml




namespace boost {
namespace serialization {

template<typename T>
struct tracking_level< kml::linear<T> > {
	typedef mpl::integral_c_tag tag;
	typedef mpl::int_<track_never> type;
	BOOST_STATIC_CONSTANT(
	int,
	value = tracking_level::type::value
	);
};

} // namespace serialization
} // namespace boost



#endif

