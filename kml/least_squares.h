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

#ifndef LEAST_SQUARES_H
#define LEAST_SQUARES_H


#include <boost/numeric/bindings/traits/ublas_matrix.hpp>
#include <boost/numeric/bindings/traits/ublas_symmetric.hpp>
#include <boost/numeric/bindings/traits/ublas_vector.hpp>

#include <boost/numeric/bindings/lapack/posv.hpp>
#include <boost/numeric/bindings/lapack/ppsv.hpp>

#include <boost/progress.hpp>


namespace ublas = boost::numeric::ublas;
//namespace atlas = boost::numeric::bindings::atlas;
namespace lapack = boost::numeric::bindings::lapack;



/*! least squares
	solve Ax=b, where A is a positive definite matrix

	\ingroup helper
*/

template<typename MatT, typename VecT>
void least_squares( MatT const &A, VecT &x, VecT const &b ) {

    // define a matrix type (flexible for future changes)
    // has to be column major, otherwise lapack bindings can choke on it
    typedef ublas::matrix<double, ublas::column_major> matrix_type;
    typedef ublas::symmetric_matrix<double, ublas::lower, ublas::column_major> symmetric_type;
    
    // assert some stuff
    assert( x.size() == b.size() );
    assert( A.size1() == x.size() );
    
    // copy the matrix, needed because it is also a work space matrix
    symmetric_type symm_work( A );

    // the rhs (right-hand-side) matrix has to be at least as large as the number of columns in H, because the
    // result of lapack::gelsd will be placed there. Therefore, rhs has to be as "tall" as the
    // weight vector
    matrix_type B( x.size(), 1 );
    ublas::matrix_column<matrix_type> B_col( B, 0 );
    B_col.assign( b );

    // perform the actual least squares fit
    lapack::ppsv( symm_work, B );

    // copy the answer to the weight vector
    x.assign( B_col );
}






/*! perform ridge regression
	solve (A+I*\lambda)x=b, where A is a positive definite matrix

	\ingroup helper
*/

template<typename MatT, typename VecT>
void ridge_regression( MatT const &A, VecT &x, VecT const &b, typename MatT::value_type const &lambda ) {

    // define a matrix type (flexible for future changes)
    // has to be column major, otherwise lapack bindings can choke on it
    typedef typename MatT::value_type value_type;
    typedef ublas::matrix<value_type, ublas::column_major> matrix_type;
    typedef ublas::symmetric_matrix<value_type, ublas::lower, ublas::column_major> symmetric_type;

    // all dimensions must be equal
    assert( x.size() == b.size() );
    assert( A.size1() == x.size() );
    
    // create matrix, only 0.5N^2 memory will be used.
    // add the small value /lambda to the diagonal of the work matrix
    symmetric_type symm_work( A );
    for( unsigned int i=0; i<symm_work.size1(); ++i )
        symm_work(i,i) += lambda;

    // the rhs (right-hand-side) matrix has to be at least as large as the number of columns in H, because the
    // result of lapack::gelsd will be placed there. Therefore, rhs has to be as "tall" as the
    // weight vector
    matrix_type B( x.size(), 1 );
    ublas::matrix_column<matrix_type> B_col( B, 0 );
    B_col.assign( b );

    // perform the actual least squares (or least norm) fit
    lapack::ppsv( symm_work, B );

    // copy the answer to the weight vector
    x.assign( B_col );

}


/*! perform roughness penalty regression
	solve (A+lambda*L)x=b

	\ingroup helper
*/

template<typename MatT, typename VecT>
void roughness_penalty_regression( MatT const &A, MatT const &L, VecT &x, VecT const &b, typename MatT::value_type const &lambda ) {

    // define a matrix type (flexible for future changes)
    // has to be column major, otherwise lapack bindings can choke on it
    typedef typename MatT::value_type value_type;
    typedef ublas::matrix<value_type, ublas::column_major> matrix_type;
    typedef ublas::symmetric_matrix<value_type, ublas::lower, ublas::column_major> symmetric_type;

    // all dimensions must be equal
    assert( x.size() == b.size() );
    assert( A.size1() == x.size() );
    
    // create matrix, only 0.5N^2 memory will be used.
    // add the small value /lambda to the diagonal of the work matrix
    symmetric_type symm_work( A + lambda * L );

    // the rhs (right-hand-side) matrix has to be at least as large as the number of columns in H, because the
    // result of lapack::gelsd will be placed there. Therefore, rhs has to be as "tall" as the
    // weight vector
    matrix_type B( x.size(), 1 );
    ublas::matrix_column<matrix_type> B_col( B, 0 );
    B_col.assign( b );

    // perform the actual least squares (or least norm) fit
    lapack::ppsv( symm_work, B );

    // copy the answer to the weight vector
    x.assign( B_col );

}









template<typename MatT, typename VecT>
void generalised_least_squares( MatT const &A, MatT const &B, VecT const &c, VecT const &d, VecT &x ) {
}


















/*
 
old code:
 
 
    // create a view in the matrix (everything except the intercept), and copy the symmetric
    // kernel matrix into that. Also, create a vector view of the intercept term, and assign it 1's
    // (atlas::set does not work in this case)
    ublas::matrix_range<matrix_type> H_part( H, ublas::range(0,H.size1()), ublas::range(1, H.size2()) );
    ublas::matrix_column<matrix_type> intercept( H, 0 );
    H_part.assign( K );
    std::fill( intercept.begin(), intercept.end(), 1.0 );
 
    // the rhs (right-hand-side) matrix has to be at least as large as the number of columns in H, because the
    // result of lapack::gelsd will be placed there. Therefore, rhs has to be as "tall" as the
    // weight vector
    matrix_type rhs( weight_vector.size(), 1 );
    ublas::matrix_column<matrix_type> rhs_col(rhs, 0);
    ublas::vector_range<ublas::matrix_column<matrix_type> > rhs_view( rhs_col, ublas::range(0, targets.size()) );
    rhs_view.assign( targets );
 
    // perform the actual least squares fit
    lapack::gelsd( H, rhs );
    
    // copy the answer to the weight vector
    weight_vector.assign( rhs_col );
 
 
 
*/












#endif

