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

#ifndef KERNELS_H
#define KERNELS_H

// Boost uBLAS includes
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/symmetric.hpp>


// Boost numeric bindings to e.g. ATLAS
#include <boost/numeric/bindings/traits/ublas_matrix.hpp>
#include <boost/numeric/bindings/traits/ublas_symmetric.hpp>
#include <boost/numeric/bindings/traits/ublas_vector.hpp>
#include <boost/numeric/bindings/traits/std_vector.hpp>
#include <boost/numeric/bindings/atlas/cblas.hpp>

//#include <boost/numeric/bindings/lapack/gelsd.hpp>

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>

#include <boost/type_traits.hpp>
#include <boost/static_assert.hpp>
#include <boost/progress.hpp>

#include <boost/function.hpp>

#include <boost/tuple/tuple.hpp>


#include <cmath>
#include <iostream>
#include <vector>

#include <kml/detail/math.hpp>




// use like this:
//   Gaussian<2> kernel( 1.0 );
//   Gaussian<4> kernel (1.0);
//   double inner_prod( a, b, kernel );

//   gramm_matrix( X, X, kernel );



namespace ublas = boost::numeric::ublas;
namespace atlas = boost::numeric::bindings::atlas;
//namespace lapack = boost::numeric::bindings::lapack;

enum kernel_type { Gaussian_trait, polynomial_trait };
enum precomp_type { in_prod, sq_diff };







// Prune indices from some index vector, given a weight vector.
// Indices will be pruned if weights are within the prune limit.
// Weights will be cleared, i.e. set to zero exactly.
// perhaps this can be done smarter, however, as it is, it is quite optimal
// the longer series (indices vector) is copied to itself, and the "write head"
// will only make progress if the index remains. So, the "read head" will in case
// of deletion be ahead of its write counterpart. At the end, the remainder
// is erased.
template<typename VecI, typename VecV>
inline void prune_indices( VecI &indices, VecV &values ) {
    typedef typename VecV::value_type value_type;
    typename VecI::iterator it_read = indices.begin();
    typename VecI::iterator it_write = indices.begin();
    while( it_read != indices.end() ) {
        if ( kml::detail::is_zero( values[*it_read] ) )
            values[ *it_read++ ] = 0.0;
        else
            (*it_write++) = (*it_read++);
    }
    indices.erase( it_write, it_read );
}


// TO MOVE

template<typename Type>
inline Type const square( Type const &value ) {
    return value * value;
}


template<typename matX, typename matR>
inline void outer_prod( matX const &X, matR &result ) {
    if (X.size1()>X.size2()) {
        for (int col=0; col<X.size2(); ++col) {
            ublas::matrix_column<matX const> X_col( X, col );
            atlas::spr( X_col, result );
        }
    } else {
        for (int row=0; row<X.size1(); ++row) {
            ublas::matrix_row<matX const> X_row( X, row );
            atlas::spr( X_row, result );
        }
    }
}


template<typename MatX, typename MatR>
inline void inner_prod( MatX const &X, MatR &result ) {
    MatX temp( X.size2(), X.size2() );
    atlas::gemm<MatX>( ublas::trans(X), X, temp );
    result.assign( temp );
}

template<typename MatX, typename MatR>
inline void inner_prod( MatX const &X, MatX const &Y, MatR &result ) {
    MatX temp( X.size2(), X.size2() );
    atlas::gemm<MatX>( trans(X), Y, temp );
    result.assign( temp );
}


template<typename matX, typename matR>
inline void outer_norm2( matX const &X, matR &result ) {

    typedef typename matX::value_type Type;

    // result will contain pairwise ||x_i-x_j||^2 of all elements of X
    // use the identity ||u-v||^2 == (u-v)^T * (u-v) = u^T*u - 2 * u^T*v + v^T*v

    // start with the ordinary inner product kernel matrix call
    outer_prod( X, result );
    //kernel_precomp<Type, in_prod>::kernel_matrix( X, result );

    // get the diagonal from result filled with self-inner products
    // create a vector filled with a constant
    ublas::vector<Type> diag(ublas::matrix_vector_range<ublas::symmetric_matrix<Type> >(result, ublas::range(0, result.size1()), ublas::range(0, result.size1())));
    ublas::vector<Type> one_vector( result.size1() );
    atlas::set
        ( 1.0, one_vector );

    // multiply the result, for the -2 in "u^T*u - 2 * u^T*v + v^T*v"
    // do this after the diagonal vector was assigned!
    result *= -2.0;

    // spr2: A <- x yT + y xT + A,  A symmetric real matrix in packed format
    atlas::spr2( one_vector, diag, result );
}





template<typename MatT>
void preserved_resize( MatT &M, typename MatT::size_type size1, typename MatT::size_type size2 ) {
    MatT temp( M );
    M.resize( size1, size2, false );
    //std::cout << "M is now of size " << M.size1() << " x " << M.size2() << std::endl;

    unsigned int range_size1 = std::min(temp.size1(),M.size1());
    unsigned int range_size2 = std::min(temp.size2(),M.size2());

    ublas::matrix_range<MatT> temp_view( temp, ublas::range(0,range_size1), ublas::range(0,range_size2));
    ublas::matrix_range<MatT> M_view( M, ublas::range(0,range_size1), ublas::range(0,range_size2));
    M_view.assign( temp_view );

    //std::cout << "resize complete " << std::endl;

}

template<typename VecT>
void preserved_resize( VecT &v, typename VecT::size_type size ) {
    assert( size > v.size() );
    VecT temp( v );
    v.resize( size );
    std::copy( temp.begin(), temp.end(), v.begin() );
}



// shrink underlying matrix by eliminating the indexed columns and rows
template<typename MatT>
void preserved_shrink( MatT &X, typename MatT::size_type index1, typename MatT::size_type index2 ) {
    MatT temp( X.size1()-1, X.size2()-1 );

    typedef typename MatT::size_type size_type;

    // copy first part, and second part
    // NOTE extremely inefficient workaround
    size_type row_t=0;
    for( size_type row=0; row<index1; ++row ) {
        size_type col_t=0;
        for( size_type col=0; col<index2; ++col ) {
            temp(row_t,col_t++) = X(row,col);
        }
        for( size_type col=index2+1; col<X.size2(); ++col ) {
            temp(row_t,col_t++) = X(row,col);
        }
        row_t++;
    }
    for( size_type row=index1+1; row<X.size1(); ++row ) {
        size_type col_t=0;
        for( size_type col=0; col<index2; ++col ) {
            temp(row_t,col_t++) = X(row,col);
        }
        for( size_type col=index2+1; col<X.size2(); ++col ) {
            temp(row_t,col_t++) = X(row,col);
        }
        row_t++;
    }

    X.resize( X.size1()-1, X.size2()-1 );
    X.assign( temp );
}


template<typename Matrix>
void remove_column( Matrix &X, typename Matrix::size_type index ) {

	typedef typename Matrix::size_type size_type;

	for( size_type i=index; i<(X.size2()-1); ++i ) {
		ublas::column(X,i).assign( ublas::column(X,i+1));
	}
	X.resize( X.size1(), X.size2()-1);
}



// shrink underlying vector by eliminating the index from the vector
template<typename VecT>
void preserved_shrink( VecT &v, typename VecT::size_type index ) {
    VecT temp( v.size()-1 );

    typedef typename VecT::size_type size_type;

    // copy first part, and second part
    // NOTE extremely inefficient workaround
    size_type j=0;
    for( size_type i=0; i<index; ++i ) {
        temp[j++] = v[i];
    }
    for( size_type i=index+1; i<v.size(); ++i ) {
        temp[j++] = v[i];
    }
    v.resize( v.size()-1 );
    std::copy( temp.begin(), temp.end(), v.begin() );
}










template<int order>
class order_weight {
public:
static const int value = (order == 0 ? 1 : 0);
};




template<typename Type, precomp_type>
class kernel_precomp {
public:
    inline static Type inner_prod( std::vector<Type> u, std::vector<Type> v );
};


template<typename Type>
class kernel_precomp<Type, in_prod> {
public:
    inline static Type inner_prod( std::vector<Type> u, std::vector<Type> v ) {
        // return the classic inner product of these two vectors
        return atlas::dot( u, v );
    }


    inline static void kernel_matrix( ublas::matrix<Type> const &X, ublas::symmetric_matrix<Type> &result ) {
        // compute X * X^T, that's all there's to it here

        // if the number of rows is larger than the number of columns, iterate over columns,
        // else iterate over rows
        if (X.size1()>X.size2()) {
            for (int col=0; col<X.size2(); ++col) {
                ublas::matrix_column<ublas::matrix<Type> const> X_col( X, col );
                atlas::spr( X_col, result );
            }
        } else {
            for (int row=0; row<X.size1(); ++row) {
                ublas::matrix_row<ublas::matrix<Type> const> X_row( X, row );
                atlas::spr( X_row, result );
            }
        }
    }
};

template<typename Type>
class kernel_precomp<Type, sq_diff> {
public:
    inline static Type inner_prod( std::vector<Type> u, std::vector<Type> v ) {
        // this is kind of not the best way?? Costs a lot of memory, and is inefficient..
        // I can pass on the temporary vector in this case... (from the calling function)
        // ...


        std::vector<Type> temp(u);
        atlas::axpy(-1.0,v,temp);
        return atlas::dot( temp, temp );
    }

    inline static void kernel_matrix( ublas::matrix<Type> const &X, ublas::symmetric_matrix<Type> &result ) {}
}
;









template<typename Type, typename KernelType>
Type inner_prod( std::vector<Type> const &u, std::vector<Type> const &v, KernelType const &kernel ) {
    // well, do this rather efficient & short, etc..
    return kernel( kernel_precomp<Type, KernelType::precomp_trait>::inner_prod( u, v ) );
}




//
// compute the kernel matrix given the data in X and an instantiated kernel function
//

template<typename Type, typename KernelType>
void kernel_matrix( ublas::matrix<Type> const &X, KernelType const &kernel, ublas::symmetric_matrix<Type> &result ) {

    // the kernel matrix is symmetric
    result.resize( X.size1(), X.size1() );
    result.clear();

    // precompute a symmetric matrix with either inner products or squared norm2's
    kernel_precomp<Type, KernelType::precomp_trait>::kernel_matrix( X, result );

    // apply the kernel function to all (1/2)*N^2 resulting matrix elements
    for (unsigned int row=0; row < result.size1(); ++row)
        for (unsigned int col=0; col <= row; ++col)
            result(row,col) = kernel( result(row,col) );

    // result is a correct kernel matrix of N by N
}



// compute a generic gram matrix

// inputs: matrix expressions, outputs: matrix expressions....

template<typename MatT, typename Kernel>
void kernel_matrix( MatT const &X, MatT const &Y, Kernel const&k, MatT &result ) {

    // result <- t(X %*% t(Y))  with the inner product operator redefined
    // result(i,j) <- inner_prod( X[j,], Y[i,] )
    // result will be of size M by N, with M=rows(Y), and N=rows(X)
    // this will result in a kernel matrix if X==Y


    result.resize( Y.size1(), X.size1() );
    for( typename MatT::size_type i = 0; i < Y.size1(); ++i ) {
        for( typename MatT::size_type j = 0; j < X.size1(); ++j ) {
            ublas::matrix_row<MatT const> x_j( X, j );
            ublas::matrix_row<MatT const> y_i( Y, i );
            result( i, j ) = k( x_j, y_i );
        }
    }
}

template<typename MatT, typename VecT, typename KF>
void kernel_matrix( MatT const &X, VecT const &idx, KF const&k, ublas::matrix_range<MatT> &result ) {

    // result <- t(X %*% t(Y))  with the inner product operator redefined
    // result(i,j) <- inner_prod( X[j,], Y[i,] )
    // result will be of size M by N, with M=rows(Y), and N=rows(X)
    // this will result in a kernel matrix if X==Y

    // result must already be of the proper size!!
    for( unsigned int i = 0; i < idx.size(); ++i ) {
        for( unsigned int j = 0; j < idx.size(); ++j ) {
            ublas::matrix_row<MatT const> x_i( X, idx[i] );
            ublas::matrix_row<MatT const> x_j( X, idx[j] );
            result( i, j ) = k( x_i, x_j );
        }
    }
}


template<typename MatT, typename VecT, typename KF>
void kernel_matrix( MatT const &X, VecT const &idxX, VecT const &idxY, KF const&k, ublas::matrix_range<MatT> &result ) {

    // result <- t(X %*% t(Y))  with the inner product operator redefined
    // result(i,j) <- inner_prod( X[j,], Y[i,] )
    // result will be of size M by N, with M=rows(Y), and N=rows(X)
    // this will result in a kernel matrix if X==Y

    // result must already be of the proper size!!
    for( unsigned int i = 0; i < idxY.size(); ++i ) {
        for( unsigned int j = 0; j < idxX.size(); ++j ) {
            ublas::matrix_row<MatT const> x_j( X, idxX[j] );
            ublas::matrix_row<MatT const> y_i( X, idxY[i] );
	    result( i, j ) = k( x_j, y_i );
        }
    }
}








template<typename MatT, typename VecY, typename VecR, typename Kernel>
void kernel_matrix( MatT const &X, VecY const &y, Kernel const&k, VecR &result ) {

    // result <- t(X %*% t(Y))  with the inner product operator redefined
    // result(i,j) <- inner_prod( X[j,], Y[i,] )
    // result will be of size M by N, with M=rows(Y), and N=rows(X)
    // this will result in a kernel matrix if X==Y

    // sizes have to match!!
    assert( result.size() == X.size1() );
    //result.resize( X.size1() );
    for( typename MatT::size_type j = 0; j < X.size1(); ++j ) {
        ublas::matrix_row<MatT const> x_j( X, j );
        result( j ) = k( x_j, y );
    }
}


template<typename MatT, typename VecY, typename VecI, typename VecR, typename Kernel>
void kernel_matrix( MatT const &X, VecY const &y, VecI const &idx, Kernel const&k, VecR &result ) {

    // result <- t(X %*% t(Y))  with the inner product operator redefined
    // result(i,j) <- inner_prod( X[j,], Y[i,] )
    // result will be of size M by N, with M=rows(Y), and N=rows(X)
    // this will result in a kernel matrix if X==Y

    for( unsigned int i=0; i<idx.size(); ++i ) {
        ublas::matrix_row<MatT const> x_i( X, idx[i] );
        result( i ) = k( x_i, y );
    }
}











// TODO matrix type independance

template<typename MatT, typename MatR, typename Kernel>
void design_matrix( MatT const &X, MatT const &Y, Kernel const&k, MatR &result ) {

    // result <- t(X %*% t(Y))  with the inner product operator redefined
    // result(i,j) <- inner_prod( X[j,], Y[i,] )
    // result will be of size M by N, with M=rows(Y), and N=rows(X)
    // this will result in a kernel matrix if X==Y

    result.resize( Y.size1(), X.size1() + 1 );
    //ublas::matrix_column<MatR> intercept( result, 0 );

    // depends on the derivative order of the kernel type...
    atlas::set( static_cast<typename MatT::value_type>(order_weight<Kernel::order>::value), ublas::column( result, 0 ) );
    for( typename MatT::size_type i = 0; i < Y.size1(); ++i ) {
        for( typename MatT::size_type j = 0; j < X.size1(); ++j ) {
            ublas::matrix_row<MatT const> x_j( X, j );
            ublas::matrix_row<MatT const> y_i( Y, i );
            result( i, j+1 ) = k( x_j, y_i );
        }
    }
}

















// input: kernel matrix K. Output: inner product of design matrix t(H) %*% H, in packed symmetric form.
// additional memory requirements: matrix of size K.
// the non-symmetric case of H is much more straightforward, isn't it?

template<typename Type>
void packed_HtH( ublas::matrix<Type> const &H, ublas::symmetric_matrix<Type> &HtH ) {
    // compute H^T * H

    // create a copy of the kernel matrix K (needed)
    ublas::matrix<Type> temp( H );
    ublas::symmetric_adaptor<ublas::matrix<Type> > temp_symm(temp);

    // compute it's own inner product, K^T K
    atlas::syrk( CblasTrans, 1.0, temp, 0.0, temp_symm );

    // resize H^T H to the appropriate size (nr of columns of K+1) (although K is a symmetric matrix here,
    // it doesn't really matter, but it does when K is not symmetric)
    HtH.resize( H.size2()+1, H.size2()+1 );

    // take everything but the first column and row
    ublas::matrix_range<ublas::symmetric_matrix<Type> > HtH_KtK_part( HtH, ublas::range(1,HtH.size1()), ublas::range(1, HtH.size2()) );

    // copy the right parts of K^T K to the H^T H matrix
    HtH_KtK_part.assign( temp_symm );

    // as a final step: incorporate the intercept-column!
    ublas::vector<Type> one_vector( H.size1() );
    ublas::vector<Type> first_row_and_col( HtH.size1() );
    ublas::vector_range<ublas::vector<Type> > part_of_froc( first_row_and_col, ublas::range(1, first_row_and_col.size()) );
    atlas::set
        ( 1.0, one_vector );

    // a part of this is a simple matrix-vector multiplication.
    atlas::spmv( H, one_vector, part_of_froc );

    // inner product of the intercept term column equals its length, and that length
    // is the number of rows in K (which equals the number of rows in H).
    first_row_and_col(0) = H.size1();

    // copy the result to H^T H
    ublas::matrix_column<ublas::symmetric_matrix<Type> > HtH_column( HtH, 0 );
    HtH_column.assign( first_row_and_col );

    // CORRECT & VERIFIED!!
    //    std::cout << "HtH is " << HtH << std::endl;

}


template<typename Type, typename vector_type>
void packed_Hty( ublas::symmetric_matrix<Type> const &K, ublas::vector<Type> const &y, vector_type &result ) {
    result.resize( K.size2()+1 );

    // handle the intercept term
    ublas::vector<Type> one_vector( K.size1() );
    atlas::set
        ( 1.0, one_vector );
    result(0) = atlas::dot( one_vector, y );

    // compute first part: Hty <- K y (K is symmetric, so doesn't matter)
    // then second part: y <- 1 y + y
    ublas::vector_range<ublas::vector<Type> > part_of_Hty( result, ublas::range(1, result.size()) );
    atlas::spmv( K, y, part_of_Hty );
}


template<typename MatT, typename kernel_type, typename vector_type>
void memory_Hty( MatT const &X, MatT const &Y, kernel_type const &k, vector_type const &y, vector_type result ) {


    // TODO: should an intercept term be used?

    // compute memory preserving t(H) %*% y
    result.clear();
    vector_type H_col( X.size1() );
    for( typename MatT::size_type j = 0; j < X.size1(); ++j ) {
        for( typename MatT::size_type i = 0; i < Y.size1(); ++i ) {
            ublas::matrix_row<MatT const> x_j( X, j );
            ublas::matrix_row<MatT const> y_i( Y, i );
            H_col( j ) = k( x_j, y_i );
        }
        atlas::axpy( y(j), H_col, result );
    }
    // now the result should contain t(H) %*% y

}











template<typename Type, typename vector_type>
Type variance_yHw( ublas::symmetric_matrix<Type> const &K, ublas::vector<Type> const &weight_vector, vector_type const &targets ) {

    // compute ||y-Halpha||^2
    assert( K.size1() == targets.size() );
    assert( (K.size2() + 1) == weight_vector.size() );
    ublas::vector<Type> residuals( targets.size() );

    // process the intercept term
    atlas::set
        ( -weight_vector(0), residuals );
    atlas::xpy( targets, residuals );

    // the second part: y - K alpha
    // spmv(a,A,x,b,y): y <- a A x + b y, A symmetric real matrix in packed format
    // skip weight[0] or alpha_0, a.k.a. the intercept term
    ublas::vector_range<ublas::vector<Type> const> part_of_weights( weight_vector, ublas::range(1, weight_vector.size()) );
    atlas::spmv( -1.0, K, part_of_weights, 1.0, residuals );

    // done!.
    double result = atlas::nrm2( residuals );
    return( result * result );
}



// the residual sum of squares is this... residual_sum_of_squares...
// compute the sum of squares, take an intercept term into account
// sum_of_squares <- ||Ax-b||^2

// TODO: correctness check.

template<typename MatA, typename VecA, typename VecB>
typename MatA::value_type residual_sum_squares( MatA const &A, VecA const &x, std::vector<unsigned int> const &indices, VecB const &b ) {

    assert( A.size1() == b.size() );
    assert( A.size2() == x.size() );

    VecA residuals( b.size() );
    atlas::copy( b, residuals );
    atlas::scal( -1.0, residuals );
    
    //VecA residuals( -b );

    // only use the columns referenced by indices
    for( std::vector<unsigned int>::const_iterator i=indices.begin(); i!=indices.end(); ++i ) {
        //ublas::matrix_column<MatA const> A_col( A, *i );
        // TODO reintroduce atlas
	ublas::noalias(residuals) += x[*i] * ublas::column(A, *i);
	//atlas::axpy( x[*i], A_col, residuals );
    }

    // done!
    typename MatA::value_type result = atlas::dot( residuals, residuals );
    return result;
}



// in this version, the VecA doesn't contain zeros, i.e. isn't of full length
template<typename MatA, typename VecA, typename VecB>
typename MatA::value_type residual_sum_of_squares_2( MatA const &A, VecA const &x, std::vector<unsigned int> const &indices, VecB const &b ) {

    assert( A.size1() == b.size() );
    assert( x.size() == indices.size() );

    VecA residuals( -b );

    // only use the columns referenced by indices
    for( unsigned int i=0; i<indices.size(); ++i ) {
        atlas::axpy( x[i], ublas::column( A, indices[i] ), residuals );
    }

    return atlas::dot( residuals, residuals );//result;
}








#endif

