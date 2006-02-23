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

#ifndef GEMM_HPP
#define GEMM_HPP

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


struct atlas_gemm {
	template<typename MatrixA, typename MatrixX, typename MatrixB>
	static void compute( MatrixA const &A, MatrixX const &X, MatrixB &B ) {
		std::cout << "ATLAS gemm..." << std::endl;
		atlas::gemm( A, X, B );
	}
};

struct exotic_gemm {
	template<typename MatrixA, typename MatrixX, typename MatrixB>
	static void compute( MatrixA const &A, MatrixX const &X, MatrixB &B ) {
		// assume value type is defined in the matrix
		typedef typename MatrixB::value_type value_type;
		std::cout << "exotic gemm..." << std::endl;

		// indexed for now ... 
		for( unsigned int row_A = 0; row_A < A.size1(); ++row_A ) {
			for( unsigned int col_X = 0; col_X < X.size2(); ++col_X ) {
				value_type result = scale_or_dot( A(row_A,0), X(0,col_X) );
				for (unsigned int col_A = 1; col_A < A.size2(); ++col_A )
					result += scale_or_dot( A(row_A,col_A), X(col_A,col_X) );
				B( row_A, col_X ) = result;
			}
		}
	}
};




/*! 
Matrix A has to be a matrix that is 

*/

template<typename MatrixA, typename MatrixX, typename MatrixB>
void gemm( MatrixA const &A, MatrixX const &X, MatrixB &B ) {
	// figure out what to do (during compile-time)
	// this is not completely correct now, should be as in the scale_or_dot
	mpl::if_< boost::is_scalar<typename MatrixA::value_type>, atlas_gemm, exotic_gemm >::type::compute( A, X, B );
}



}} // namespace kml::detail

#endif



