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

#ifndef INCOMPLETE_CHOLESKY_HPP
#define INCOMPLETE_CHOLESKY_HPP

#include <kml/matrix_view.hpp>
#include <boost/utility.hpp>

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>

#include <boost/numeric/bindings/atlas/cblas.hpp>
#include <boost/property_map.hpp>

/*!
 
This algorithm will construct a partial QR decomposition, similar to the 
the incomplete cholesky decomposition described in [1--2]. 
 
The naming of matrices is a bit different, because of a geometric interpretation
of the algorithm. It is identical to an orthogonalisation procedure where the
furthest vector is added to the basis. Additional documentation is a todo.
 
 
Efficient SVM training using low-rank kernel representations
Fine, Scheinberg
http://jmlr.csail.mit.edu/papers/volume2/fine01a/fine01a.pdf
Figure 2, Page 255
 
Predictive low-rank decomposition for kernel methods
Bach, Jordan
http://cmm.ensmp.fr/~bach/bach_jordan_csi.pdf
Section 2.1
 
*/


namespace ublas = boost::numeric::ublas;
namespace atlas = boost::numeric::bindings::atlas;


namespace kml {




template<typename Kernel, typename PropertyMap>
class incomplete_cholesky {
public:

    typedef Kernel kernel_type;
    typedef PropertyMap map_type;
    typedef typename boost::property_traits<PropertyMap>::key_type key_type;
    typedef typename boost::property_traits<PropertyMap>::value_type object_type;

    incomplete_cholesky( typename boost::call_traits<Kernel>::param_type k,
                         typename boost::call_traits<PropertyMap>::param_type map ):
    kernel_function(k), data(&map) {}



    // if needed...
    void compute_Q() {

        // after an orthogonalisation has finished, one may use this routine to compute the Q part
        // however, this only works for linear kernels, and, may be not as precise as the result of
        // an householder-based QR decomposition
        //

        Q.resize( (*data)[pivot[0]].get<0>().size(), RT.size2() );
        Q.clear();

        // data model: column n of A is equal to data[n], A(1:m,i) = data[i]
        // A has n columns, vectors of length m
        // Q is m by n, with m the length of the vectos in A

        ublas::column( Q, 0 ) = (*data)[pivot[0]].get<0>() / RT.matrix(0,0);
        for( int i=1; i < basis_size; ++i ) {
            typedef ublas::matrix_vector_slice< ublas::matrix<double> > range_type;
            range_type mvr( RT.matrix, ublas::slice( i, 0, i ), ublas::slice( 0, 1, i ) );
            ublas::column( Q, i ) = ( (*data)[pivot[i]].get<0>() -
                                      ublas::prod( ublas::subrange( Q, 0, Q.size1(), 0, i ), mvr ) ) / RT.matrix(i,i);
        }
    }



    void add_to_basis( int index ) {

        // move some stuff around
        std::swap( pivot[ basis_size ], pivot[ index ] );
        std::swap( squared_distance[ basis_size ],  squared_distance[ index ] );
        RT.swap_row( basis_size, index );

        // resize the R matrix
        RT.grow_column();

        double perpendicular_distance = std::sqrt( squared_distance[ basis_size ] );

        // lookup the current key
        key_type current_key = pivot[ basis_size ];

        // for convenience... clear the first part
        for( int row = 0; row < basis_size; ++row ) {
            RT.matrix( row, basis_size ) = 0.0;
        }
        RT.matrix( basis_size, basis_size ) = perpendicular_distance;
        if ( basis_size == 0 ) {
            for( int row = basis_size+1; row < pivot.size(); ++row ) {
                RT.matrix( row, basis_size ) = kernel_function((*data)[current_key].get<0>(),(*data)[pivot[row]].get<0>())
                                               / perpendicular_distance;
                squared_distance[ row ] -= RT.matrix( row, basis_size ) * RT.matrix( row, basis_size );
            }
        } else {
            // setup slices and views, and fire it through ATLAS
            typedef ublas::matrix_vector_slice<ublas::matrix<double> > slice_type;
            ublas::matrix_range< ublas::matrix<double> > RT_range( ublas::subrange( RT.matrix, basis_size+1, RT.size1(), 0, basis_size  ));
            slice_type basis_range( RT.matrix,
                                    ublas::slice(basis_size, 0, basis_size),
                                    ublas::slice(0, 1, basis_size) );
            ublas::vector<double> target( RT.size1()-basis_size-1 );

            atlas::gemv( RT_range, basis_range, target );

            for( int row = basis_size+1; row < pivot.size(); ++row ) {
                RT.matrix( row, basis_size ) = ( kernel_function((*data)[current_key].get<0>(),(*data)[pivot[row]].get<0>())
                                                 - target[row-basis_size-1] )
                                               / perpendicular_distance;
                squared_distance[ row ] -= RT.matrix( row, basis_size ) * RT.matrix( row, basis_size );
            }
        }

        squared_distance[ basis_size ] = 0.0;
        ++basis_size;
    }



    // rank could leave in the future
    template<typename KeyIterator>
    void compute_R( KeyIterator begin, KeyIterator end, int rank ) {

        // make sure we have a pivot table
        std::copy( begin, end, back_inserter( pivot ) );

        // initialise the R matrix
        RT.reserve( pivot.size(), 0 );

        // the order of our data should not have changed since the std::copy, three statements ago
        for( typename std::vector<key_type>::iterator i = pivot.begin(); i != pivot.end(); ++i ) {
            squared_distance.push_back( kernel_function((*data)[*i].get<0>(),(*data)[*i].get<0>()) );
        }

        double max_distance = 1.0;
        // set basis size to 0
        basis_size = 0;

        // only a small number...
        // TODO: figure out a suitable stopping criterion
        //while( basis_size < rank ) {
        while( max_distance > 1e-6 ) {
            //while( (remaining_set.size() > 0) && ( max_pdsq > 1e-6 ) ) {

            max_distance = 0.0;
            int migrate_index = 0;

            for( int i = basis_size; i < squared_distance.size(); ++i ) {
                if ( squared_distance[i] > max_distance ) {
                    max_distance = squared_distance[i];
                    migrate_index = i;
                }
            }

            if ( max_distance > 1e-6 ) {
                add_to_basis( migrate_index );
            } else {
                //std::cout << "Basis could not be extended!" << std::endl;
            }

        }

        //std::cout << "Orthogonalisation finished, size of basis:  " << basis_size << std::endl;
    }


    // this is the key lookup table; serves for pretty much everything
    std::vector< key_type > pivot;
    std::vector<double> squared_distance;

    int basis_size;

    // QR decomposition, upper triangular matrix R, transposed
    kml::matrix_view< ublas::matrix<double> > RT;

    // QR decomposition, orthogonal matrix Q
    ublas::matrix<double> Q;

    // data and the kernel
    kernel_type kernel_function;
    PropertyMap const *data;

}
; // class incomplete cholesky



} // namespace kml

#endif

