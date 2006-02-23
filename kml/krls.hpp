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

#ifndef KRLS_HPP
#define KRLS_HPP

#include <boost/numeric/bindings/traits/ublas_matrix.hpp>
#include <boost/numeric/bindings/traits/ublas_symmetric.hpp>
#include <boost/numeric/bindings/traits/ublas_vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/bindings/atlas/cblas.hpp>
#include <kml/kernel_machine.hpp>
#include <kml/regression.hpp>
#include <kml/symmetric_view.hpp>


namespace atlas = boost::numeric::bindings::atlas;


namespace kml {


/*!
\brief Kernel Recursive Least Squares
   
A preprocessing step is taken to absorb the bias term into the weight vector w, by
redefining w as (w^T, b)^T and \f$\phi\f$ as (phi^T,1)^T. For details, see Engel et al,
Sparse Online Greedy Support Vector Regression, below on page 3:
\f$k(x,x')=k(x,x') + \lambda^2\f$, where \f$\lambda\f$ is a small, positive constant.
 
\section bibliography References
-# Engel et al., 2003. Kernel Recursive Least Squares. 
*/


template< typename Problem, typename Kernel, typename PropertyMap, class Enable = void>
class krls: public kernel_machine< Problem, Kernel, PropertyMap > {}
;



template< typename Problem, typename Kernel, typename PropertyMap >
class krls< Problem, Kernel, PropertyMap, typename boost::enable_if< is_regression<Problem> >::type>:
    public kernel_machine< Problem, Kernel, PropertyMap > {

    typedef kernel_machine< Problem, Kernel, PropertyMap > base_type;
    typedef typename base_type::kernel_type kernel_type;
    typedef double scalar_type;

    typedef symmetric_view< ublas::matrix<double> > symmetric_type;

    typedef ublas::matrix<double> matrix_type;
    typedef ublas::vector<double> vector_type;

    typedef typename Problem::input_type input_type;
    typedef typename Problem::output_type output_type;

    typedef typename boost::property_traits<PropertyMap>::key_type key_type;
    typedef typename boost::property_traits<PropertyMap>::value_type object_type;

public:
    krls( scalar_type n, scalar_type l,
          typename boost::call_traits<kernel_type>::param_type k,
          typename boost::call_traits<PropertyMap>::param_type map ):
    base_type(k,map), nu(n), lambda_squared(l*l) {}

    template< typename TokenIterator >
    krls( TokenIterator const begin, TokenIterator const end,
          typename boost::call_traits<kernel_type>::param_type k,
          typename boost::call_traits<PropertyMap>::param_type map ):
    base_type(k,map) {
        nu = 1e-1;						// default value
        scalar_type lambda=1e-3;		// default value
        TokenIterator token(begin);
        if ( token != end ) {
            nu = boost::lexical_cast<double>( *token++ );
            if ( token != end )
                lambda = boost::lexical_cast<double>( *token );
        }
        lambda_squared=lambda*lambda;
    }


    output_type operator()( input_type const &x ) {

        vector_type temp_K( base_type::key_lookup.size() );
        base_type::fill_kernel( x, base_type::key_lookup.begin(), base_type::key_lookup.end(), temp_K.begin() );
        for( int i=0; i < temp_K.size(); ++i )
            temp_K[i] += lambda_squared;
        return atlas::dot( base_type::weight, temp_K );

    }


    /*! learn the entire range of keys indicated by this range */
    template<typename KeyIterator>
    void learn( KeyIterator begin, KeyIterator end ) {
        KeyIterator key_iterator(begin);
        while( key_iterator != end ) {
            increment( *key_iterator );
            ++key_iterator;
        }
    }


    /*! \param key key of example in data */
    void increment( key_type const &key ) {

        //std::cout << "running key " << key << " through KRLS... " << std::endl;

        // calculate the base_type::kernel function on (x_t,x_t), needed later on
        scalar_type k_tt = base_type::kernel( key, key ) + lambda_squared;

        // check whether dictionary is still not initialised
        if ( base_type::key_lookup.empty() ) {

            // there is no dictionary yet, so initialise all variables
            // resize the matrix K, its inverse R and matrix P to 1 x 1
            K.grow_row_column();
            R.grow_row_column();
            P.grow_row_column();

            // and use values as stated in the paper
            K.matrix(0,0) = k_tt;
            R.matrix(0,0) = 1.0 / k_tt;
            P.matrix(0,0) = 1.0;

            // add to weight vector
            base_type::weight.push_back( (*base_type::data)[key].get<1>() / k_tt );

            // add to support vector set
            base_type::key_lookup.push_back( key );

        } else {

            // KRLS already initialised, continue
            vector_type a_t( K.size1() );
            vector_type k_t( K.size1() );

            // fill vector k_t
            base_type::fill_kernel( key, base_type::key_lookup.begin(), base_type::key_lookup.end(), k_t.begin() );
            for( typename vector_type::size_type i=0; i<k_t.size(); ++i )
                k_t[i] += lambda_squared;

            // a_t <- R %*% k_t
            ublas::matrix_range< ublas::matrix<double> > R_range( R.view() );
            ublas::symmetric_adaptor< ublas::matrix_range< ublas::matrix<double> > > R_view( R_range );
            atlas::symv( R_view, k_t, a_t );
            scalar_type delta_t = k_tt - atlas::dot( k_t, a_t );

            // Perform Approximate Linear Dependency (ALD) test
            if (delta_t > nu) {

                // add x_t to support vector set, adjust all needed variables
                unsigned int old_size = base_type::key_lookup.size();

                // update K (equation 14)
                // fetch a view into the last row of the matrix of the _old_ size
                K.grow_row_column();
                ublas::matrix_vector_slice< ublas::matrix<double> > K_row_part( K.shrinked_row(old_size) );
                atlas::copy( k_t, K_row_part );
                K.matrix( old_size, old_size ) = k_tt;

                // update R (equation 14)
                scalar_type factor = static_cast<scalar_type>(1) / delta_t;
                atlas::syr( factor, a_t, R_view );
                R.grow_row_column();
                ublas::matrix_vector_slice< ublas::matrix<double> > R_row_part( R.shrinked_row(old_size) );
                atlas::scal( -factor, a_t );
                R_row_part.assign( a_t );
                R.matrix( old_size, old_size ) = factor;

                // update P (equation 15)
                // assign unit vector with 1 on last element.
                P.grow_row_column();
                ublas::matrix_vector_slice< ublas::matrix<double> > P_row_part( P.shrinked_row(old_size) );
                atlas::set
                    (  0.0, P_row_part );
                P.matrix( old_size, old_size ) = 1.0;

                // adjust weight vector alpha (equation 16)
                factor = (*base_type::data)[key].get<1>() - atlas::dot(k_t,base_type::weight);
                atlas::axpy( factor, a_t, base_type::weight );

                // add new weight to the weight vector
                base_type::weight.push_back( factor / delta_t );

                // add support vector to the set
                base_type::key_lookup.push_back( key );

            } else {
                // support vector set unchanged (see algorithmic on page 4 of paper)
                // adjust weight vector and permutation matrix P
                // P_a <- P_t-1 %*% a_t
                vector_type P_a( base_type::key_lookup.size() );

                // spmv(A,x,y)       y <- A x
                ublas::matrix_range< ublas::matrix<double> > P_range( P.view() );
                ublas::symmetric_adaptor< ublas::matrix_range< ublas::matrix<double> > > P_view( P_range );
                atlas::symv( P_view, a_t, P_a );

                // 1 / (1 + a_t %*% P_(t-1) %*% a)
                scalar_type factor = 1.0 / (1.0 + atlas::dot( a_t, P_a ));

                // update weights (equation 13)
                atlas::symv( factor* (*base_type::data)[key].get<1>() - atlas::dot(k_t,base_type::weight),
                             R_view, P_a, static_cast<scalar_type>(1), base_type::weight );

                // update permutation matrix (equation 14)
                atlas::syr( -factor, P_a, P_view );
            }
        }
    }


private:

    symmetric_type K;                // kernel matrix K
    symmetric_type R;                // inverse of kernel matrix K
    symmetric_type P;                // permutation matrix P
    scalar_type nu;                  // ALD parameter
    scalar_type lambda_squared;      // kernel function addition
};



} // namespace kml

#endif
