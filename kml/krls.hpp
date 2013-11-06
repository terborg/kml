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

#ifndef KML_KRLS_HPP
#define KML_KRLS_HPP

#include <boost/numeric/bindings/ublas/matrix.hpp>
#include <boost/numeric/bindings/ublas/matrix_proxy.hpp>
#include <boost/numeric/bindings/ublas/symmetric.hpp>
#include <boost/numeric/bindings/ublas/vector.hpp>
#include <boost/numeric/bindings/blas.hpp>
#include <kml/kernel_machine.hpp>
#include <kml/regression.hpp>
#include <kml/symmetric_view.hpp>

#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/tracking.hpp>
#include <boost/serialization/vector.hpp>


namespace blas = boost::numeric::bindings::blas;


namespace kml {


/*!
\brief Kernel Recursive Least Squares
   
A preprocessing step is taken to absorb the bias term into the weight vector w, by
redefining w as (w^T, b)^T and \f$\phi\f$ as (phi^T,1)^T. For details, see Engel et al,
Sparse Online Greedy Support Vector Regression, below on page 3:
\f$k(x,x')=k(x,x') + \lambda^2\f$, where \f$\lambda\f$ is a small, positive constant.

Kernel matrix K is actually not needed by the algorithm and is therefore dropped.
 
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

    typedef ublas::matrix<double, ublas::column_major> matrix_type;
    typedef ublas::vector<double> vector_type;
    typedef symmetric_view< matrix_type > symmetric_type;


    typedef typename Problem::input_type input_type;
    typedef typename Problem::output_type output_type;

    typedef typename boost::property_traits<PropertyMap>::key_type key_type;
    typedef typename boost::property_traits<PropertyMap>::value_type object_type;


    friend class boost::serialization::access;

public:
    krls( scalar_type tol, scalar_type l,
          typename boost::call_traits<kernel_type>::param_type k,
          typename boost::call_traits<PropertyMap>::param_type map ):
    base_type(k,map), tolerance(tol), lambda_squared(l*l) {}

    template< typename TokenIterator >
    krls( TokenIterator const begin, TokenIterator const end,
          typename boost::call_traits<kernel_type>::param_type k,
          typename boost::call_traits<PropertyMap>::param_type map ):
    base_type(k,map) {
        tolerance = 1e-1;			// default value
        scalar_type lambda=1e-3;		// default value
        TokenIterator token(begin);
        if ( token != end ) {
            tolerance = boost::lexical_cast<double>( *token++ );
            if ( token != end )
                lambda = boost::lexical_cast<double>( *token );
        }
        lambda_squared=lambda*lambda;
    }


    output_type operator()( input_type const &x ) {

        vector_type temp_K( basis_key.size() );
        this->fill_kernel( x, basis_key.begin(), basis_key.end(), temp_K.begin() );
        for( unsigned int i=0; i < temp_K.size(); ++i )
            temp_K[i] += lambda_squared;
        return blas::dot( weight, temp_K );

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

        // calculate the base_type::kernel function on (x_t,x_t), needed later on
        scalar_type k_tt = this->kernel( key, key ) + lambda_squared;

        // check whether dictionary is still not initialised
        if ( basis_key.empty() ) {

            // there is no dictionary yet, so initialise all variables
            // resize the inverse matrix R, and matrix P to 1 x 1
            R.grow_row_column();
            P.grow_row_column();

            // and use values as stated in the paper
            R.matrix(0,0) = 1.0 / k_tt;
            P.matrix(0,0) = 1.0;

            // add to weight vector
            weight.push_back( this->output(key) / k_tt );

            // add to support vector set
            basis_key.push_back( key );

        } else {

            // KRLS already initialised, continue
            vector_type a_t( basis_key.size() );
            vector_type k_t( basis_key.size() );

            // fill vector k_t
            this->fill_kernel( key, basis_key.begin(), basis_key.end(), k_t.begin() );
            for( typename vector_type::size_type i=0; i<k_t.size(); ++i )
                k_t[i] += lambda_squared;

            // a_t <- R %*% k_t
            ublas::matrix_range< matrix_type > R_range( R.view() );
            ublas::symmetric_adaptor< ublas::matrix_range< matrix_type > > R_view( R_range );
            blas::symv( 1.0, R_view, k_t, 1.0, a_t );
            scalar_type delta_t = k_tt - blas::dot( k_t, a_t );

            // Perform Approximate Linear Dependency (ALD) test
            // If the ALD is larger than some previously set tolerance, add the input to 
            // the dictionary (or basis vector set)
            if (delta_t > tolerance) {

                // add x_t to support vector set, adjust all needed variables
                std::size_t old_size = basis_key.size();

                // update inverse matrix R (equation 14)
                scalar_type factor = static_cast<scalar_type>(1) / delta_t;
                blas::syr( factor, a_t, R_view );
                R.grow_row_column();
                ublas::matrix_vector_slice< matrix_type > R_row_part( R.shrinked_row(old_size) );
                blas::scal( -factor, a_t );
                R_row_part.assign( a_t );
                R.matrix( old_size, old_size ) = factor;

                // update permutation matrix P (equation 15)
                // assign unit vector with 1 on last element.
                P.grow_row_column();
                ublas::matrix_vector_slice< matrix_type > P_row_part( P.shrinked_row(old_size) );
                blas::set(  0.0, P_row_part );
                P.matrix( old_size, old_size ) = 1.0;

                // adjust weight vector alpha (equation 16)
                factor = this->output(key) - blas::dot(k_t,weight);
                blas::axpy( factor, a_t, weight );

                // add new weight to the weight vector
                weight.push_back( factor / delta_t );

                // add support vector to the set
                basis_key.push_back( key );

            } else {
                // support vector set unchanged (see algorithmic on page 4 of paper)
                // adjust weight vector and permutation matrix P
                // P_a <- P_t-1 %*% a_t
                vector_type P_a( basis_key.size() );

                // spmv(A,x,y)       y <- A x
                ublas::matrix_range< matrix_type > P_range( P.view() );
                ublas::symmetric_adaptor< ublas::matrix_range< matrix_type > > P_view( P_range );
                blas::symv( 1.0, P_view, a_t, 1.0, P_a );

                // 1 / (1 + a_t %*% P_(t-1) %*% a_t)
                scalar_type factor = 1.0 / (1.0 + blas::dot( a_t, P_a ));

                // update weights (equation 13)
                blas::symv( factor* this->output(key) - blas::dot(k_t,weight),
                             R_view, P_a, static_cast<scalar_type>(1), weight );

                // update permutation matrix (equation 14)
                blas::syr( -factor, P_a, P_view );
            }
        }
    }

    // loading and saving capabilities
    template<class Archive>
    void serialize( Archive &archive, unsigned int const version ) {
        archive & boost::serialization::base_object<base_type>(*this);
        archive & tolerance;
        archive & lambda_squared;
        archive & basis_key;
        archive & weight;
        archive & R;
        archive & P;
    }

private:
    scalar_type tolerance;               // ALD tolerance parameter
    scalar_type lambda_squared;          // kernel function addition
    std::vector< key_type > basis_key;   // a vector containing basis vector keys
    std::vector< scalar_type > weight;   // weights associated with the basis vectors
    symmetric_type R;                    // inverse of kernel matrix K
    symmetric_type P;                    // permutation matrix P
};

} // namespace kml






namespace boost {
namespace serialization {

template< typename Problem, typename Kernel, typename PropertyMap, typename Enable >
struct tracking_level< kml::krls<Problem,Kernel,PropertyMap,Enable> > {
    typedef mpl::integral_c_tag tag;
    typedef mpl::int_<track_never> type;
    BOOST_STATIC_CONSTANT(
        int,
        value = tracking_level::type::value
    );
};

} // namespace serialization
} // namespace boost


#endif


