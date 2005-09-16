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
		// in this case: 
                // - Matrix A has ublas::vectors as elements
		// - Vector x has scalars
		// - Vector b has ublas::vectors as elements

		// TODO
		// if x has vectors, then b will have scalars (I will use an inner_prod for that).

		std::cout << "exotic gemv..." << std::endl;

		// indexed for now ...
		for( unsigned int m = 0; m < A.size1(); ++m ) {
			typename VectorB::value_type result = zero_element( A(0,0) );
			for( unsigned int n = 0; n < A.size2(); ++n )
				result += A(m,n) * x[n];
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



