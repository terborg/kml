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

#ifndef GEMV_HPP
#define GEMV_HPP

#include <boost/range/begin.hpp>
#include <boost/range/end.hpp>
#include <boost/range/size.hpp>
#include <boost/range/value_type.hpp>
#include <boost/mpl/if.hpp>
#include <boost/type_traits/is_scalar.hpp>

#include <boost/numeric/bindings/atlas/cblas.hpp>
#include <boost/numeric/bindings/traits/ublas_vector.hpp>
#include <boost/numeric/bindings/traits/ublas_matrix.hpp>

#include <kml/detail/zero_element.hpp>
#include <kml/detail/scale_or_dot.hpp>

#include <boost/call_traits.hpp>

#include <iostream>

namespace atlas = boost::numeric::bindings::atlas;
namespace mpl = boost::mpl;


namespace kml { namespace detail {


struct atlas_gemv {
	template<typename Matrix, typename VectorX, typename VectorB>
	static void compute( Matrix const &A, VectorX const &x, VectorB &b ) {
		std::cout << "ATLAS gemv..." << std::endl;
		atlas::gemv( A, x, b );
	}
};

struct exotic_gemv {
	template<typename Matrix, typename VectorX, typename VectorB>
	static void compute( Matrix const &A, VectorX const &x, VectorB &b ) {
		typedef typename boost::range_value<VectorB>::type value_type;
		std::cout << "exotic gemv..." << std::endl;

		// indexed for now ...
		for( unsigned int m = 0; m < A.size1(); ++m ) {
			value_type result = scale_or_dot( A(m,0), x[0] );
			for( unsigned int n = 1; n < A.size2(); ++n )
				result += scale_or_dot( A(m,n), x[n] );
			b[m] = result;
		}
	}
};




/*! 
Matrix A has to be a matrix that is 

*/

template<typename Matrix, typename VectorX, typename VectorB>
void gemv( Matrix const &A, VectorX const &x, VectorB &b ) {
	// figure out what to do (during compile-time)
	mpl::if_< boost::is_scalar<typename Matrix::value_type>, atlas_gemv, exotic_gemv >::type::compute( A, x, b );
}



}} // namespace kml::detail

#endif



