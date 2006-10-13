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

#ifndef MATRIX_VIEW_HPP
#define MATRIX_VIEW_HPP

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/bindings/atlas/cblas.hpp>

#include <boost/serialization/ublas_matrix.hpp>

#include <boost/serialization/access.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/tracking.hpp>




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
    friend class boost::serialization::access;
    typedef ublas::vector<double> vector_type;
    typedef M matrix_type;
    typedef typename M::size_type size_type;

    matrix_view() {
        // 1 by 1, for fast growth determination later on (i.e. use size * 2)
        matrix.resize(1,1,false);
        view_rows=0;
        view_cols=0;
    }


    unsigned int find_pow2_neighbour( unsigned int n ) {
	unsigned int answer = 1;
	while ( n > 0 ) {
		answer <<= 1;
		n >>= 1;
	}
	return answer;
    }

    // if you have some pre-knowledge, or want to pre-reserve memory
    // does not preserve the matrix!!
    void reserve( size_type rows, size_type cols ) {
	view_rows = rows;
	view_cols = cols;
        matrix.resize( find_pow2_neighbour(rows), find_pow2_neighbour(cols), false );
    }


    /*! Increase the number of columns in the matrix view by one.


    */
    void grow_column() {
        if ( view_cols == matrix.size2() )
            matrix.resize( matrix.size1(), matrix.size2() << 1 );
        ++view_cols;
    }

    void shrink_column() {
	--view_cols;
    }

    void grow_row() {
        if ( view_rows == matrix.size1() )
            matrix.resize( matrix.size1() << 1, matrix.size2() );
        ++view_rows;
    }

    void shrink_row() {
	--view_rows;
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
    void swap_remove_column( size_type index ) {
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

    void swap_row( int index1, int index2 ) {
        ublas::matrix_range<M> matrix_view(matrix, ublas::range(0,view_rows),ublas::range(0,view_cols));
        ublas::matrix_row< ublas::matrix_range<M> > index1_row( matrix_view, index1 );
        ublas::matrix_row< ublas::matrix_range<M> > index2_row( matrix_view, index2 );
	atlas::swap( index1_row, index2_row );
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
    inline ublas::matrix_range<M> const shrinked_view() const {
        return ublas::matrix_range<M>(matrix, ublas::range(0,view_rows-1),ublas::range(0,view_cols-1));
    }

    // also provide a shrunken column view (view_rows x view_cols-1)
    inline ublas::matrix_range<M> const shrinked_column_view() const {
        return ublas::matrix_range<M>(matrix,ublas::range(0,view_rows),ublas::range(0,view_cols-1));
    }


    inline ublas::matrix_vector_slice<M> const row( int nr ) {
        return ublas::matrix_vector_slice<M>( matrix, ublas::slice( nr, 0, view_cols ), ublas::slice( 0, 1, view_cols ) );
    }

    //     inline ublas::matrix_row< ublas::matrix_range<M> > const row( int nr ) {
    //         return ublas::row( ublas::matrix_range<M>(matrix, ublas::range(0,view_rows),ublas::range(0,view_cols)), nr );
    //     }

    inline ublas::matrix_row< ublas::matrix_range<M> > const shrinked_row( int nr ) {
        return ublas::row( ublas::matrix_range<M>(matrix, ublas::range(0,view_rows),ublas::range(0,view_cols-1)), nr );
    }

    inline ublas::matrix_column< ublas::matrix_range<M> > const column( int nr ) {
        return ublas::column( ublas::matrix_range<M>(matrix, ublas::range(0,view_rows),ublas::range(0,view_cols)), nr );
    }

    inline ublas::matrix_column< ublas::matrix_range<M> > const shrinked_column( int nr ) {
        return ublas::column( ublas::matrix_range<M>(matrix, ublas::range(0,view_rows-1),ublas::range(0,view_cols)), nr );
    }


    // loading and saving capabilities
    template<class Archive>
    void save( Archive &archive, unsigned int const version ) const {
        // make a copy of the view and save that part only
        M matrix_copy( ublas::subrange(matrix, 0, view_rows, 0, view_cols ));
        archive << matrix_copy;
    }

    template<class Archive>
    void load( Archive &archive, unsigned int const version ) {
        // restore the copy of the view
        archive >> matrix;

        // store the view size
        view_rows = matrix.size1();
        view_cols = matrix.size2();

        // find the value of the MSB of the view_rows, that multiplied by 2 should equal pow2_rows
        unsigned int find_pow2_rows = view_rows;
        unsigned int pow2_rows = 1;
        while( find_pow2_rows > 0 ) {
            find_pow2_rows >>= 1;
            pow2_rows <<= 1;
        }

        // find the value of the MSB of the view_cols, that multiplied by 2 should equal pow2_cols
        unsigned int find_pow2_cols = view_cols;
        unsigned int pow2_cols = 1;
        while( find_pow2_cols > 0 ) {
            find_pow2_cols >>= 1;
            pow2_cols <<= 1;
        }

        // (preserved) resize of the matrix to sizes that are powers of 2
        matrix.resize( pow2_rows, pow2_cols, true );
    }

    // a convenience macro to auto-create the serialize() member function
    BOOST_SERIALIZATION_SPLIT_MEMBER()

    // internal memory is a (dense) UBLAS matrix
    M matrix;
private:
    size_type view_rows;
    size_type view_cols;
};


} // namespace kml






namespace boost {
namespace serialization {

template<typename T>
struct tracking_level< kml::matrix_view<T> > {
    typedef mpl::integral_c_tag tag;
    typedef mpl::int_<track_never> type;
    BOOST_STATIC_CONSTANT(
        int,
        value = tracking_level::type::value
    );
};

}
}

#endif

