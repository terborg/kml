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

#ifndef DESIGN_MATRIX_HPP
#define DESIGN_MATRIX_HPP

#include <boost/numeric/bindings/blas.hpp>
#include <boost/numeric/bindings/ublas/matrix.hpp>
#include <boost/numeric/bindings/ublas/matrix_proxy.hpp>
#include <boost/numeric/bindings/ublas/vector.hpp>
#include <boost/range/begin.hpp>
#include <boost/range/empty.hpp>
#include <boost/range/end.hpp>
#include <boost/range/size.hpp>
#include <boost/range/value_type.hpp>


/*!
 
Design matrix construction algorithm. Mostly referred to as matrix H.
 
Source is an existing support vector set, which could be the output of a pre-processing step. However,
mostly this is equal to target. Target are all input data points associated with the fitting problem.
 
Changeable design matrix as well (can be later extended with data buffering, exchanging, etc.)
 
Design matrix:
- Sources
- Targets
 
Often, sources are a subset of the targets, but not necessarily always. It is quite a safe assumption to 
say that sources and targets are two seperate sets.
 
Actions:
- A point is added to the source set: a column is added to the design matrix
- A point is added to the target set: a row is added to the design matrix
 
For efficiency reasons, we do want the storage to be continuous, use matrix proxies
in conjunction with ATLAS, and resizes should only take place once in a while.

*/


namespace blas = boost::numeric::bindings::blas;
namespace ublas = boost::numeric::ublas;


// TODO fix value types etc.

namespace kml {

template<class Range, class Kernel>
void design_matrix( Range const &source, Range const &target, Kernel const &kernel,
                    double const bias_value, ublas::matrix<double> &result ) {

    // FIXME the result should be resized already??
    // do a non-preserving resize
    result.resize( boost::size(target), boost::size(source)+1, false );

    // TODO check blas additional functionality
    ublas::matrix_column<ublas::matrix<double> > bias_col( result, 0 );
    //blas::set( bias_value, bias_col );
    std::fill( boost::begin(bias_col), boost::end(bias_col), bias_value );

    for( unsigned int s = 0; s < boost::size( source ); ++s ) {
        for( unsigned int t = 0; t < boost::size( target ); ++t ) {
            // or is [] usable already as well?
            result( t, s+1 ) = kernel( *(boost::begin(source) + s),
                                       *(boost::begin(target) + t) );
        }
    }
}


} // namespace kml



#endif

