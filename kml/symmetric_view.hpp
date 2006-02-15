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

#ifndef SYMMETRIC_VIEW_HPP
#define SYMMETRIC_VIEW_HPP

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/symmetric.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/bindings/atlas/cblas.hpp>


namespace ublas = boost::numeric::ublas;

namespace kml {


/*!
\brief A symmetric view of a dense matrix.
\param M underlying dense matrix class

This class provides a view of a symmetric matrix, and performs efficient memory management 
by cutting down the number of calls to .resize() of the underlying matrix. 

If we consider a matrix that grows by one row or column at a time, 
a naive resize strategy would cost O(N) memory allocation calls. It will become quadratically 
more expensive to copy the content of the matrix as it grows in size.
Because of this, memory management is done in a similar way as that of std::vector.
If needed, the size of the underlying matrix is doubled in the number of rows or in the number of columns.
This takes O(log N) memory allocation calls only, delivering a considerable speedup over the naive approach.

It is assumed that the symmetric view changes by 1 row and 1 column simultaneously at a time, for which member
functions are available.

It can be directly used in conjunction with calls to atlas bindings, as shown below.

\code
kml::symmetric_view< ublas::matrix<double> > A;    // view in matrix A
ublas::vector<double> x;                        // vector x
ublas::vector<double> b;                        // result vector b
atlas::symv( A.view(), x, b );                  // compute b <- Ax
\endcode

*/

template<class M>
class symmetric_view {
public:
    typedef ublas::vector<double> vector_type;
    typedef M matrix_type;
    typedef typename M::size_type size_type;

    symmetric_view() {
        // 1 by 1, for fast growth determination later on (i.e. use size * 2)
        matrix.resize(1,1,false);
        view_rows=0;
        view_cols=0;
    }

    // if you have some pre-knowledge, or want to pre-reserve memory
    // does not preserve the matrix!!
    void reserve( size_type rows, size_type cols ) {
        matrix.resize( std::max(rows,1), std::max(cols,1), false );
    }

    /*! Increase the number of columns and the number of rows in the symmetric view by one.
    */    
    void grow_row_column() {
        if ( view_rows == matrix.size1() )
            matrix.resize( matrix.size1() << 1, matrix.size2() << 1 );
        ++view_rows;
        ++view_cols;
    }

    /*! Remove a column from the current matrix view at a given column index.
        \param index the n'th column, first column is located at 0.
	
        Keep the order of columns as they are, so all columns have to be "moved to the left". This is not 
	the most efficient strategy in removing a column, 
     */
    void remove_column( int index ) {
        ublas::matrix_range<M> matrix_view(matrix, ublas::range(0,view_rows),ublas::range(0,view_cols));
        for( size_type i=index; i<(view_cols-1); ++i ) {
            ublas::matrix_column< ublas::matrix_range<M> > cur_col( matrix_view, i);
            ublas::matrix_column< ublas::matrix_range<M> > next_col( matrix_view, i+1);
            atlas::copy( next_col, cur_col );
        }
        --view_cols;
    }

    // copies the last column to index and decreases the view_cols
    void swap_remove_column( int index ) {
        ublas::matrix_range<M> matrix_view(matrix, ublas::range(0,view_rows),ublas::range(0,view_cols));
        ublas::matrix_column< ublas::matrix_range<M> > last_col( matrix_view, view_cols-1);
        ublas::matrix_column< ublas::matrix_range<M> > index_col( matrix_view, index );
	atlas::copy( last_col, index_col );
	--view_cols;
    }


    // NOTE shouldn't this be done with atlas::copy as well?
    void remove_row( size_type index ) {
        ublas::matrix_range<M> matrix_view(matrix, ublas::range(0,view_rows),ublas::range(0,view_cols));
        for( size_type i=index; i<(view_rows-1); ++i ) {
            ublas::row(matrix_view,i).assign( ublas::row(matrix_view,i+1));
        }
        --view_rows;
    }

    // copies the last row to index and decreases the view_rows
    void swap_remove_row( size_type index ) {
	ublas::matrix_range<M> matrix_view(matrix, ublas::range(0,view_rows),ublas::range(0,view_cols));
        ublas::matrix_row< ublas::matrix_range<M> > last_row( matrix_view, view_rows-1);
        ublas::matrix_row< ublas::matrix_range<M> > index_row( matrix_view, index);
	atlas::copy( last_row, index_row );
	--view_rows;
    }

    void remove_row_col( size_type index ) {
        remove_column(index);
        remove_row(index);
    }

    // work for: row_major matrix with a symmetric view adaptor (row major as well)
    // warning: sensitive code
    void swap_remove_row_col( size_type index ) {
	ublas::matrix_range<M> matrix_view(matrix, ublas::range(0,view_rows),ublas::range(0,view_cols));
        size_type index_last = view_rows-1;

	// perhaps this is an overkill for 1 atlas-op.
	// perhaps can be done with 2 matrix vector slices?
	ublas::matrix_row< ublas::matrix_range<M> > last_row( matrix_view, index_last );
        ublas::matrix_row< ublas::matrix_range<M> > index_row( matrix_view, index );
	ublas::vector_range< ublas::matrix_row< ublas::matrix_range<M> > > last_row_part_1( last_row, ublas::range(0,index) );
	ublas::vector_range< ublas::matrix_row< ublas::matrix_range<M> > > index_row_part_1( index_row, ublas::range(0,index) );
	atlas::copy( last_row_part_1, index_row_part_1 );

	// and a difficult element copy, in same direction. can be atlasified
	// note that index_last is one less than the matrix size
	for( size_type i=index+1; i<index_last; ++i ) {
		matrix( i, index ) = matrix( index_last, i );
	}
	
	// index_last,index_last -> index,index
	matrix( index, index ) = matrix( index_last, index_last );
	
	--view_rows;
	--view_cols;
    }

    void flush() {
        matrix.resize( view_rows, view_cols, true );
    }

    size_type const size1() const {
        return view_rows;
    }

    size_type const size2() const {
        return view_cols;
    }

    inline ublas::matrix_range<M> const view() {
        return ublas::matrix_range<M>(matrix, ublas::range(0,view_rows),ublas::range(0,view_cols));
    }

    // also provide shrinked views: this is very useful for the inverted matrix updates
    // i.e. the matrix is resized first, and then an algorithm computes the updated inverse
    inline ublas::matrix_range<M> const shrinked_view() {
        return ublas::matrix_range<M>(matrix, ublas::range(0,view_rows-1),ublas::range(0,view_cols-1));
    }

    inline ublas::matrix_row< ublas::matrix_range<M> > const row( size_type const nr ) {
        return ublas::row( ublas::matrix_range<M>(matrix, ublas::range(0,view_rows),ublas::range(0,view_cols)), nr );
    }

    
    inline ublas::matrix_vector_slice<M> const shrinked_row( size_type const nr ) {
    	return ublas::matrix_vector_slice<M>( matrix, ublas::slice( nr, 0, view_cols-1 ), ublas::slice( 0, 1, view_cols-1 ) );
    }
    
//     inline ublas::matrix_row< ublas::matrix_range<M> > const shrinked_row( size_type const nr ) {
//         return ublas::row( ublas::matrix_range<M>(matrix, ublas::range(0,view_rows),ublas::range(0,view_cols-1)), nr );
//     }

    inline ublas::matrix_column< ublas::matrix_range<M> > const column( size_type const nr ) {
        return ublas::column( ublas::matrix_range<M>(matrix, ublas::range(0,view_rows),ublas::range(0,view_cols)), nr );
    }

    inline ublas::matrix_column< ublas::matrix_range<M> > const shrinked_column( size_type const nr ) {
        return ublas::column( ublas::matrix_range<M>(matrix, ublas::range(0,view_rows-1),ublas::range(0,view_cols)), nr );
    }

/*
	code snippets from several routines:

RVM:
                atlas::syr( sigma_ii * beta * beta, row(work_mat,i), sigma_symm );
                preserved_resize( sigma, new_size, new_size );
                ublas::matrix_vector_slice<ublas::matrix<double, ublas::column_major> > sigma_row_part( sigma, ublas::slice(old_size,0,old_size), ublas::slice(0,1,old_size));
                noalias(sigma_row_part) = -beta * beta * sigma_ii * row(work_mat,i);
                sigma(old_size,old_size) = sigma_ii;

	
AOSVR:		
	    -->>> factor = 1/K_tt + k_t^T * delta
	    
	    atlas::spr( factor, delta, R );
            preserved_resize( R, new_size, new_size );
            ublas::matrix_vector_slice<symmetric_type> R_row_part( R, ublas::slice(old_size,0,old_size),
                    ublas::slice(0,1,old_size) );
            R_row_part.assign( factor * delta );
            R( old_size, old_size ) = factor;
	
	
KRLS:

	    --->>> factor = 1/delta_t
	    delta_t = k_tt - atlas::dot( k_t, a_t );
	    
                atlas::spr( factor, a_t, K_inv );
                preserved_resize( K_inv, new_size, new_size );
                ublas::matrix_vector_slice<symmetric_type> K_inv_row_part( K_inv, ublas::slice(old_size,0,old_size),
                        ublas::slice(0,1,old_size) );
                atlas::scal( -factor, a_t );
                K_inv_row_part.assign( a_t );
                K_inv( old_size, old_size ) = factor;


*/


    // grow the inverse of this matrix. Needed ingredients:
    // k_tt  = kernel( x_t, x_t ).
    // delta = current sv set projected on current inv. matrix
    // k_t   = kernel( x, x_t ) new point against all current svs
    // scale = easy way to control a precomputed scale of the inverse matrix (e.g. -1.0)
    template<class VectorA, class VectorB>
    bool grow_inverse(  double k_tt,
                        VectorB const &k_t, VectorA const &delta ) {

        
/*	std::cout << "delta: " << delta << std::endl;
	std::cout << "k_t:   " << k_t << std::endl;*/
	

	if (view_rows == 1) {
	    
	    // initialise the inverse
            grow_row_column();
            matrix(0,0) = k_tt;
            matrix(1,0) = -1.0;
            matrix(1,1) = 0.0;
	    return false;
        } else {

	    // compute factor
	    // TODO make flexible, using scale (in such way also KRLS works with this)
	    double factor =  k_tt + atlas::dot( k_t, delta );
// 	    std::cout << "factor is " << factor;

      	    //std::cout << "factor is " << -1.0 / (k_tt + atlas::dot( k_t,delta)) << std::endl;

/*	    double limit = std::numeric_limits<double>::epsilon() *
                      static_cast<double>( view_rows*view_rows*view_rows);
	    std::cout << " limit  "  << limit << std::endl;*/
	    

	    // perform sanity check
//            if ( factor < (view_rows * view_rows * view_rows * 1e-6))
            if ( std::fabs(factor) < 1e-8 ) {
		// this update can't continue: numeric stability problems
                return true;
            } else {

	    	factor = -1.0 / factor;
	        
		ublas::matrix_range< ublas::matrix<double> > my_range( view() );
                ublas::symmetric_adaptor< ublas::matrix_range< ublas::matrix<double> > > my_symm_view( my_range );

		// perform a rank-1 update of the R matrix
                atlas::syr( factor, delta, my_symm_view );

		int old_size = view_cols;

                // resize the matrix
                grow_row_column();

                // fetch a view of a part of the last row
		ublas::matrix_row< ublas::matrix_range<M> > last_row_part( shrinked_row(old_size));

		// further update the matrix
		atlas::copy( delta, last_row_part );
		atlas::scal( factor, last_row_part );
		matrix(old_size, old_size) = factor;


		return false;
            }
        }
    }


    void shrink_inverse( size_type index ) {

        // remove from the inverse matrix
        if ( view_rows==2) {
            --view_rows;
            --view_cols;
            matrix(0,0)=0.0;
        } else {
      	    ublas::matrix_range< ublas::matrix<double> > my_range( view() );
            ublas::symmetric_adaptor< ublas::matrix_range< ublas::matrix<double> > > my_symm_view( my_range );

	    // take a correct row, from the symmetric adaptor!!
	    vector_type temp_row( ublas::row( my_symm_view, index ) );
            atlas::syr( -1.0/matrix(index,index), temp_row, my_symm_view );

            // NOTE removing a row_col can be done more efficiently if
            //      the matrix is symmetric
            remove_row( index );
	    remove_column( index );
        }
    }





    
    // internal memory is a matrix
    M matrix;
private:
    size_type view_rows;
    size_type view_cols;
};


} // namespace kml








#endif
