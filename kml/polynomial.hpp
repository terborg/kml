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

#ifndef POLYNOMIAL_HPP
#define POLYNOMIAL_HPP

#include <boost/call_traits.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/tracking.hpp>
#include <boost/type_traits.hpp>
#include <kml/input_value.hpp>
#include <kml/linear.hpp>

namespace kml {

/*!
 
\brief polynomial kernel
\param T defines the underlying input data type
 
This is a template class that creates a functor for the polynomial kernel.
 
\ingroup kernels
 
*/

template<typename T>
class polynomial: public std::binary_function<T,T,typename input_value<T>::type> {
public:
    typedef polynomial<T> type;
    typedef typename input_value<T>::type scalar_type;
    typedef typename mpl::int_<0>::type derivative_order;
    friend class boost::serialization::access;

    /*! Construct an uninitialised polynomial kernel */
    polynomial() {}

    /*! Refinement of CopyConstructable */
    polynomial( type const &other ) {
        copy( other );
    }

    /*! Construct a polynomial kernel by providing values for gamma, lambda, and degree
        \param gamma  the scale of the inner product
        \param lambda the bias of the inner product
        \param degree the order of the polynomial kernel  */
    polynomial( typename boost::call_traits<scalar_type>::param_type gamma,
                typename boost::call_traits<scalar_type>::param_type lambda,
                typename boost::call_traits<scalar_type>::param_type degree ): scale(gamma), bias(lambda), order(degree) {}

    /*! Construct a polynomial kernel by providing TokenIterators */
    template<typename TokenIterator>
    polynomial( TokenIterator const begin, TokenIterator const end ) {
        // set default values
        scale = 1.0;
        bias = 0.0;
        order = 3.0;
        TokenIterator token(begin);
        if ( token != end )
            scale = boost::lexical_cast<scalar_type>( *token++ );
        if ( token != end )
            bias = boost::lexical_cast<scalar_type>( *token++ );
        if ( token != end )
            order = boost::lexical_cast<scalar_type>( *token );
    }

    /*! Refinement of Assignable */
    type &operator=( type const &other ) {
        if (this != &other) {
            destroy();
            copy(other);
        }
        return *this;
    }

    /*! Returns the result of the evaluation of the polynomial kernel for these points
    	\param u input pattern u
        \param v input pattern v
    	\return \f$ (\gamma * u^T v + \lambda)^d \f$
       */
    scalar_type operator()( T const &u, T const &v ) const {
        return std::pow( scale * linear<T>()(u,v) + bias, order );
    }

    // loading and saving capabilities
    template<class Archive>
    void serialize( Archive &archive, unsigned int const version ) {
        archive & scale;
        archive & bias;
        archive & order;
    }

    // for debugging purposes
    friend std::ostream& operator<<(std::ostream &os, type const &k) {
        os << "Polynomial kernel (" << k.scale << "<u,v>+" << k.bias << ")^" << k.order << std::endl;
        return os;
    }

private:
    void copy( type const &other ) {
        scale = other.scale;
        bias = other.bias;
        order = other.order;
    }
    void destroy() {}

    scalar_type scale;
    scalar_type bias;
    scalar_type order;

};


} // namespace kml





namespace boost {
namespace serialization {

template<typename T>
struct tracking_level< kml::polynomial<T> > {
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



