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

    /*! Refinement of Assignable */
    type &operator=( type const &other ) {
		if (this != &other) { destroy(); copy(other); }
		return *this;
    }

    /*! Construct a polynomial kernel
        \param gamma  the scale of the inner product
        \param lambda the bias of the inner product
        \param d      the order of the polynomial kernel
      */
    polynomial( typename boost::call_traits<scalar_type>::param_type gamma,
                typename boost::call_traits<scalar_type>::param_type lambda,
                typename boost::call_traits<scalar_type>::param_type d): scale(gamma), bias(lambda), order(d) {}

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

#endif



