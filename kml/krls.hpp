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


#ifndef KRLS_HPP
#define KRLS_HPP

#include <online_greedy_determinate.hpp>

#include <boost/numeric/bindings/atlas/cblas.hpp>
#include "kernels.h"

#include <design_matrix.hpp>



namespace atlas = boost::numeric::bindings::atlas;


namespace kml {


/*!
\brief Kernel Recursive Least Squares
\param I the input type
\param O the output type
\param K the kernel type
   
\todo
- Perhaps this adding of a value to the kernel can be avoided if the same structure is used as is used by
the on line_svm.
- Migrate matrices to view_matrix (avoid resize costs)

A preprocessing step is taken to absorb the bias term into the weight vector w, by
redefining w as (w^T, b)^T and \f$\phi\f$ as (phi^T,1)^T. For details, see Engel et al,
Sparse Online Greedy Support Vector Regression, below on page 3:
\f$k(x,x')=k(x,x') + \lambda^2\f$, where \f$\lambda\f$ is a small, positive constant.

\section bibliography References
-# Engel et al., 2003. Kernel Recursive Least Squares. 
*/


template<typename I, typename O, template<typename,int> class K>
class krls: public online_greedy_determinate<I,O,K> {

    typedef online_greedy_determinate<I,O,K> base_type;
    typedef typename base_type::kernel_type kernel_type;
    typedef double scalar_type;
    typedef ublas::symmetric_matrix<double> symmetric_type;
    typedef ublas::matrix<double> matrix_type;
    typedef ublas::vector<double> vector_type;
    typedef I input_type;
    typedef O output_type;
    

public:
    krls( typename boost::call_traits<kernel_type>::param_type k, scalar_type n=1e-1, scalar_type l=1e-3 ):
            base_type(k), nu(n), lambda_squared(l*l) {}

    output_type operator()( input_type const &x ) {
    
        // TODO do a trick here::: replace the base_type::kernel with another functor,
	// but then one with the value added. Then, dispatch to the underlying base_type::kernel machine.
	vector_type temp_K( base_type::support_vector.size() );
        for( int i=0; i < base_type::support_vector.size(); ++i )
           temp_K[i] = base_type::kernel( base_type::support_vector[i], x ) + lambda_squared;
        
        return atlas::dot( base_type::weight, temp_K );

/*	
	// matrix_type temp_H;
        base_type::kernel_matrix( base_type::support_vector, X, base_type::kernel, temp_H );
	if (base_type::kernel.order==0) {
	for( int i = 0; i<temp_H.size1(); ++i )
            for( int j = 0; j<temp_H.size2(); ++j )
                temp_H(i,j) += lambda_squared;
	}
        atlas::gemv( temp_H, base_type::weight, result );*/
    
    }
    
    
    /*! \param x_t an input pattern
        \param y_t an output pattern */
    void push_back( input_type const &x_t, output_type const &y_t ) {

	//std::cout << "running point " << x_t << " through KRLS..." << std::endl;
    
	// calculate the base_type::kernel function on (x_t,x_t), needed later on
        scalar_type k_tt = base_type::kernel( x_t, x_t ) + lambda_squared;

        // check whether dictionary is still not initialised
        if ( base_type::support_vector.size() == 0 ) {

	    // there is no dictionary yet, so initialise all variables
            // resize the matrix H, its inverse Hinv and matrix P to 1 x 1
            H.resize(1);
            R.resize(1);
            P.resize(1);

            // and use values as stated in the paper
            H(0,0) = k_tt;
            R(0,0) = 1.0 / k_tt;
            P(0,0) = 1.0;

            // add to weight vector
	    base_type::weight.push_back( y_t / k_tt );

            // add to support vector set
            base_type::support_vector.push_back( x_t );

        } else {

            // KRLS already initialised, continue
            vector_type a_t( H.size1() );
            vector_type k_t( H.size1() );

	    // fill vector k_t
	    for( int i=0; i<base_type::support_vector.size(); ++i )
	       k_t[i] = base_type::kernel( base_type::support_vector[i], x_t ) + lambda_squared;

            atlas::spmv( R, k_t, a_t );
            scalar_type delta_t = k_tt - atlas::dot( k_t, a_t );

            // do ALD test
            if (delta_t > nu) {
                // add x_t to support vector set, adjust all needed variables
                unsigned int old_size = base_type::support_vector.size();
                unsigned int new_size = old_size + 1;

                // update H (equation 14)
                preserved_resize( H, new_size, new_size );
                ublas::matrix_vector_slice<symmetric_type> H_row_part( H, ublas::slice(old_size,0,old_size),
                        ublas::slice(0,1,old_size) );
                H_row_part.assign( k_t );
                H( old_size, old_size ) = k_tt;

                // update R (equation 14)
                scalar_type factor = static_cast<scalar_type>(1) / delta_t;
                atlas::spr( factor, a_t, R );
                preserved_resize( R, new_size, new_size );
                ublas::matrix_vector_slice<symmetric_type> R_row_part( R, ublas::slice(old_size,0,old_size),
                        ublas::slice(0,1,old_size) );
                atlas::scal( -factor, a_t );
                R_row_part.assign( a_t );
                R( old_size, old_size ) = factor;

                // update P (equation 15)
                // assign unit vector with 1 on last element.
                preserved_resize( P, new_size, new_size );
                ublas::matrix_row<symmetric_type> P_row( P, old_size );
                // NOTE TODO workaround for compiler / ublas!
                //P_row.assign_temporary( ublas::unit_vector<scalar_type>( new_size, old_size ) );
                std::fill(P_row.begin(), P_row.end(), 0.0 );
                P_row( old_size ) = 1.0;

                // adjust weight vector alpha (equation 16)
                factor = y_t-atlas::dot(k_t,base_type::weight);
                atlas::axpy( factor, a_t, base_type::weight );
		
		// add new weight to the weight vector
                base_type::weight.push_back( factor / delta_t );

                // add support vector to the set, add 1 row
	        base_type::support_vector.push_back( x_t );

            } else {
                // support vector set unchanged (see algorithmic on page 4 of paper)
                // adjust weight vector and permutation matrix P
                // P_a <- P_t-1 %*% a_t
                vector_type P_a( base_type::support_vector.size() );
                // spmv(A,x,y)       y <- A x
                atlas::spmv( P, a_t, P_a );
                // 1 / (1 + a_t %*% P_t-1 %*% a)
                scalar_type factor = 1.0 / (1.0 + atlas::dot( a_t, P_a ));

                // update base_type::weight (equation 13)
                atlas::spmv( factor*(y_t-atlas::dot(k_t,base_type::weight)), R, P_a, static_cast<scalar_type>(1), base_type::weight );

                // update permutation matrix (equation 14)
                atlas::spr( -factor, P_a, P );

            }
        }
    }





private:
    // method specific local memory
    symmetric_type H;                // base_type::kernel matrix K
    symmetric_type R;                // inverse of base_type::kernel matrix K
    symmetric_type P;                // permutation matrix P
    scalar_type nu;                   // ALD parameter
    scalar_type lambda_squared;       // base_type::kernel function addition
};



} // namespace

#endif

