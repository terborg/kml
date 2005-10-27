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

#ifndef POLYNOMIAL_HPP
#define POLYNOMIAL_HPP

#include <boost/call_traits.hpp>
#include <boost/type_traits.hpp>
#include <kml/linear.hpp>

namespace kml {

template<typename T, int N=0>
class polynomial: public std::binary_function<T,
                                              T,
                                              typename kml::power_return_type<T,N>::type> {
public:

    typedef polynomial<T,N> type;
    typedef typename boost::range_value<T>::type scalar_type;
    typedef typename mpl::int_<N>::type derivative_order;

    /*! Construct an uninitialised polynomial kernel */
    polynomial() {}

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

    friend std::ostream& operator<<(std::ostream &os, type const &k) {
	os << "Polynomial kernel (" << k.scale << "<u,v>+" << k.bias << ")^" << k.order << std::endl;
	return os;
    }

private:
    scalar_type scale;
    scalar_type bias;
    scalar_type order;

};


}

#endif

