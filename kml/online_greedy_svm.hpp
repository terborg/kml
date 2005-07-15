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


#ifndef ONLINE_GREEDY_SVM_HPP
#define ONLINE_GREEDY_SVM_HPP

#include <boost/numeric/bindings/atlas/cblas.hpp>
#include <online_greedy_determinate.hpp>

namespace kml {

/*!
\brief Sparse On-line Greedy Support Vector Regression
\param I the input type
\param O the output type
\param K the kernel type

A preprocessing step is taken to absorb the bias term into the weight vector w, by
redefining w as (w^T, b)^T and phi as (phi^T,1)^T. For details, see Engel et al,
Sparse Online Greedy Support Vector Regression, below on page 3:
k(x,x')=k(x,x') + lambda^2, where lambda is a small, positive constant.


changes to algorithm:
- removed gamma from the algorithm, and introduced the matrix inversion from KRLS
- weight is defined as alpha_star - alpha

\todo
- online greedy SVM classifier
- Optimal values of nu, nu_star can be analytically derived?


\section bibliography References
-# Yaakov Engel, Shie Mannor, and Ron Meir. Sparse On-line Greedy Support Vector Regression (SOG-SVR).
European Conference on Machine Learning (ECML) 2002, pages 84--96. 

*/



template<typename I, typename O, template<typename,int> class K>
class online_greedy_svm: public online_greedy_determinate<I,O,K> {

    typedef online_greedy_determinate<I,O,K> base_type;
    typedef typename base_type::kernel_type kernel_type;
    typedef double scalar_type;
    typedef ublas::symmetric_matrix<double> symmetric_type;
    typedef ublas::matrix<double> matrix_type;
    typedef ublas::vector<double> vector_type;
    typedef I input_type;
    typedef O output_type;

public:

    /*! 
      \param k the kernel construction parameter
      \param param_eps parameter epsilon (width of the regression tube in SVM regression)
      \param param_C parameter C (the maximum weight per support vector)
      \param n parameter nu, used by the approximate linear dependency test to check whether or not to add a
               new pattern to the support vector set. Smaller values means more support vectors.
      \param l parameter lambda. This is the constant added to each kernel function to mimick the bias term
    */
    online_greedy_svm( typename boost::call_traits<kernel_type>::param_type k,
                       scalar_type param_eps=0.1, scalar_type param_C=10.0, scalar_type n=1e-1, scalar_type l=1e-3 ):
            base_type(k), epsilon(param_eps), C(param_C), nu(n), lambda_squared(l*l) {
      // for now, set etas to 0.01
      eta = 1e-2;
      eta_star = 1e-2;
    }

    /*! \param x_t an input pattern
        \param y_t an output pattern */
    void push_back( vector_type const &x_t, scalar_type const &y_t ) {

	// calculate the base_type::kernel function on (x_t,x_t), needed later on
        scalar_type k_tt = base_type::kernel( x_t, x_t ) + lambda_squared;

	// check if init is needed
        if (base_type::support_vector.size() == 0) {

	    base_type::weight.push_back( y_t / k_tt );
	    base_type::support_vector.push_back( x_t );

            H.resize(1);
	    R.resize(1);
            H(0,0) = k_tt;
	    R(0,0) = static_cast<scalar_type>(1) / k_tt;

            AtA.resize(1);
            AtA(0,0) = static_cast<scalar_type>(1);

            Ate.resize(1);
            Ate(0) = static_cast<scalar_type>(1);

            Aty.resize(1);
            Aty(0) = y_t;

	
	} else {

            vector_type k_t( H.size1() );
            vector_type a_t( H.size1() );
            
	    for( int i=0; i<base_type::support_vector.size(); ++i )
	       k_t[i] = base_type::kernel( base_type::support_vector[i], x_t ) + lambda_squared;
	    
            atlas::spmv( R, k_t, a_t );
            scalar_type delta_t = k_tt - atlas::dot( k_t, a_t );

            // linear dependence test, check whether x_t will be added to the support vector set
            if ( delta_t > nu ) {

                unsigned int old_size = H.size1();
                unsigned int new_size = old_size + 1;

                // update matrix H
		preserved_resize( H, new_size, new_size );
                ublas::matrix_vector_slice<symmetric_type> H_row_part( H, ublas::slice(old_size,0,old_size),
                        ublas::slice(0,1,old_size) );
                H_row_part.assign( k_t );
                H( old_size, old_size ) = k_tt;

                // update matrix R (code stolen from HRLS code)
		scalar_type factor = static_cast<scalar_type>(1) / delta_t;
		atlas::spr( factor, a_t, R );
                preserved_resize( R, new_size, new_size );
                ublas::matrix_vector_slice<symmetric_type> R_row_part( R, ublas::slice(old_size,0,old_size),
                        ublas::slice(0,1,old_size) );
                atlas::scal( -factor, a_t );
                R_row_part.assign( a_t );
                R( old_size, old_size ) = factor;

                // update matrix AtA
                preserved_resize( AtA, new_size, new_size );
                ublas::matrix_vector_slice<symmetric_type> AtA_row_part( AtA, ublas::slice(old_size,0,old_size),
                        ublas::slice(0,1,old_size) );
		//AtA_row_part.clear();
		// FIXME TODO fast reset right here..
		std::for_each( AtA_row_part.begin(), AtA_row_part.end(), _1 = 0.0 );
                AtA(old_size,old_size) = 1.0;

                // FIXME
		// STL vectors? (ie use push_back)
		preserved_resize( Ate, new_size );
		preserved_resize( Aty, new_size );
                Ate( old_size ) = 1.0;
		Aty( old_size ) = y_t;
		base_type::weight.push_back( 0.0 );

                // add support vector to the set, add 1 row
                base_type::support_vector.push_back( x_t );
		
            } else {
                // dictionary remains unchanged
                atlas::xpy( a_t, Ate );
                atlas::axpy( y_t, a_t, Aty );
                atlas::spr( a_t, AtA );
            }

	    vector_type Atf( H.size1() );
	    vector_type work( H.size1() );
	    
            // Step 4. Update alpha and alpha_star
            // Atf = AtAH(alpha_star-alpha)
	    atlas::spmv( H, base_type::weight, work );
            atlas::spmv( AtA, work, Atf );
	    
	    // Update alphas
	    atlas::copy( Aty, work );
            atlas::axpy( -1.0, Atf, work );
            
            // update base_type::weight. Presume eta_star = eta, so the (-eta*+eta)epsilonAte=0,
	    // which does not need to be computed then.
	    atlas::axpy( eta_star + eta, work, base_type::weight );

	    // truncation function: u = max(-C,min(C,u)). (see end of section 4.2)
  	    for( unsigned int i=0; i<base_type::weight.size(); ++i ) {
		 base_type::weight[i] = std::max<scalar_type>(-C, std::min<scalar_type>(C, base_type::weight[i]));
  	    }
	    
        } // dictionary size > 0

    }


//     void predict( matrix_type const &X, VecT &result ) {
//         matrix_type temp_H;
//         kernel_matrix( support_vectors, X, base_type::kernel, temp_H );
// 	if (base_type::kernel.order==0) {
// 	for( int i = 0; i<temp_H.size1(); ++i )
//             for( int j = 0; j<temp_H.size2(); ++j )
//                 temp_H(i,j) += lambda_squared;
// 	}
//         atlas::gemv( temp_H, base_type::weight, result );
//     }

    
    

private:

    // configuration variables
    scalar_type lambda_squared, nu;
    scalar_type C, epsilon;
    scalar_type eta, eta_star;

    vector_type Ate;
    vector_type Aty;

    // some of these could go to the base type?
    symmetric_type H;
    symmetric_type R;
    symmetric_type AtA;
};





} // namespace kml


#endif

