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

#ifndef MATRIX_VIEW_HPP
#define MATRIX_VIEW_HPP

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/bindings/atlas/cblas.hpp>


namespace ublas = boost::numeric::ublas;
namespace atlas = ::boost::numeric::bindings::atlas;

namespace kml {


/*!
\brief A view of a dense matrix.
\param M underlying dense matrix class

This class provides a view in a dense matrix, and performs efficient memory management 
by cutting down the number of calls to .resize() of the underlying matrix. 

If we consider a matrix that grows by one row or column at a time, 
a naive resize strategy would cost O(N) memory allocation calls. It will become quadratically 
more expensive to copy the content of the matrix as it grows in size.
Because of this, memory management is done in a similar way as that of std::vector.
If needed, the size of the underlying matrix is doubled in the number of rows or in the number of columns.
This takes O(log N) memory allocation calls only, delivering a considerable speedup over the naive approach.

It is assumed matrix view changes by 1 row or 1 column (or both) at a time, for which member
functions are available.

It can be directly used in conjunction with calls to atlas bindings, as shown below.

\code
kml::matrix_view< ublas::matrix<double> > A;    // view in matrix A
ublas::vector<double> x;                        // vector x
ublas::vector<double> b;                        // result vector b
atlas::gemv( A.view(), x, b );                  // compute b <- Ax
\endcode

*/


template<class M>
class matrix_view {
public:
    typedef ublas::vector<double> vector_type;

    matrix_view() {
        // 1 by 1, for fast growth determination later on (i.e. use size * 2)
        matrix.resize(1,1,false);
        view_rows=0;
        view_cols=0;
    }

    // if you have some pre-knowledge, or want to pre-reserve memory
    // does not preserve the matrix!!
    void reserve( int rows, int cols ) {
        matrix.resize( std::max(rows,1), std::max(cols,1), false );
    }

    
    /*! Increase the number of columns in the matrix view by one.
    
    
    */    
    void grow_column() {
        if ( view_cols == matrix.size2() )
            matrix.resize( matrix.size1(), matrix.size2() << 1 );
        ++view_cols;
    }

    void grow_row() {
        if ( view_rows == matrix.size1() )
            matrix.resize( matrix.size1() << 1, matrix.size2() );
        ++view_rows;
    }

    void grow_row_column() {
        if ( view_rows == matrix.size1() )
            matrix.resize( matrix.size1() << 1, matrix.size2() << 1 );
        ++view_rows;
        ++view_cols;
    }

    /*! Remove a column from the current matrix view at a given column index.
        \param index the n'th column, first column is located at 0.
	
        This operation preserves the order of columns, which means that all columns to the 
	right of the column at index will have to be moved to the left. This is not 
	the most efficient strategy in removing a column, it will cost O(0.5NM) on average.
     */
    void remove_column( int index ) {
        ublas::matrix_range<M> matrix_view(matrix, ublas::range(0,view_rows),ublas::range(0,view_cols));
        for( int i=index; i<(view_cols-1); ++i ) {
            ublas::matrix_column< ublas::matrix_range<M> > cur_col( matrix_view, i);
            ublas::matrix_column< ublas::matrix_range<M> > next_col( matrix_view, i+1);
            atlas::copy( next_col, cur_col );
        }
        --view_cols;
    }

    
    /*! Remove a column from the current matrix view at a given column index.
        \param index the n'th column, first column is located at 0.
	
	The last column will be swapped with the column at index, after which the size of the view of the matrix
	will be reduced by 1 column. This is the most efficient way of removing a column from a dense matrix, 
	it costs O(M). 
	*/
    void swap_remove_column( int index ) {
        ublas::matrix_range<M> matrix_view(matrix, ublas::range(0,view_rows),ublas::range(0,view_cols));
        ublas::matrix_column< ublas::matrix_range<M> > last_col( matrix_view, view_cols-1);
        ublas::matrix_column< ublas::matrix_range<M> > index_col( matrix_view, index );
	atlas::copy( last_col, index_col );
	--view_cols;
    }

    /*! Remove a row from the current matrix view at a given row index.
        \param index the m'th row, first row is located at 0.
	
        This operation preserves the order of rows, so all rows have to be moved upwards. This is not 
	the most efficient strategy in removing a row, with cost of O(0.5NM) on average.
     */
    void remove_row( int index ) {
        ublas::matrix_range<M> matrix_view(matrix, ublas::range(0,view_rows),ublas::range(0,view_cols));
        for( int i=index; i<(view_rows-1); ++i ) {
            ublas::row(matrix_view,i).assign( ublas::row(matrix_view,i+1));
        }
        --view_rows;
    }

    /*! Remove a row from the current matrix view at a given row index.
        \param index the n'th row, first row is located at 0.
	
	The last row will be swapped with the row at index, after which the size of the view of the matrix
	will be reduced by one row. It costs O(N).
	*/
    void swap_remove_row( int index ) {
	ublas::matrix_range<M> matrix_view(matrix, ublas::range(0,view_rows),ublas::range(0,view_cols));
        ublas::matrix_row< ublas::matrix_range<M> > last_row( matrix_view, view_rows-1);
        ublas::matrix_row< ublas::matrix_range<M> > index_row( matrix_view, index);
	atlas::copy( last_row, index_row );
	--view_rows;
    }

    void remove_row_col( int index ) {
        remove_column(index);
        remove_row(index);
    }

    // work for: row_major matrix with a symmetric view adaptor (row major as well)
    // warning: sensitive code
    void swap_remove_row_col( int index ) {
	ublas::matrix_range<M> matrix_view(matrix, ublas::range(0,view_rows),ublas::range(0,view_cols));
        int index_last = view_rows-1;

	// perhaps this is an overkill for 1 atlas-op.
	// perhaps can be done with 2 matrix vector slices?
	ublas::matrix_row< ublas::matrix_range<M> > last_row( matrix_view, index_last );
        ublas::matrix_row< ublas::matrix_range<M> > index_row( matrix_view, index );
	ublas::vector_range< ublas::matrix_row< ublas::matrix_range<M> > > last_row_part_1( last_row, ublas::range(0,index) );
	ublas::vector_range< ublas::matrix_row< ublas::matrix_range<M> > > index_row_part_1( index_row, ublas::range(0,index) );
	atlas::copy( last_row_part_1, index_row_part_1 );

	// and a difficult element copy, in same direction. can be atlasified
	// note that index_last is one less than the matrix size
	for( int i=index+1; i<index_last; ++i ) {
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

    int size1() {
        return view_rows;
    }

    int size2() {
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

    inline ublas::matrix_row< ublas::matrix_range<M> > const row( int nr ) {
        return ublas::row( ublas::matrix_range<M>(matrix, ublas::range(0,view_rows),ublas::range(0,view_cols)), nr );
    }

    inline ublas::matrix_row< ublas::matrix_range<M> > const shrinked_row( int nr ) {
        return ublas::row( ublas::matrix_range<M>(matrix, ublas::range(0,view_rows),ublas::range(0,view_cols-1)), nr );
    }

    inline ublas::matrix_column< ublas::matrix_range<M> > const column( int nr ) {
        return ublas::column( ublas::matrix_range<M>(matrix, ublas::range(0,view_rows),ublas::range(0,view_cols)), nr );
    }

    inline ublas::matrix_column< ublas::matrix_range<M> > const shrinked_column( int nr ) {
        return ublas::column( ublas::matrix_range<M>(matrix, ublas::range(0,view_rows-1),ublas::range(0,view_cols)), nr );
    }


    // internal memory is a (dense) matrix
    M matrix;
private:
    int view_rows;
    int view_cols;
};


} // namespace kml


#endif
