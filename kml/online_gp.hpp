/***************************************************************************
 *  Sparse Online Gaussian Processes                                       *
 *  Part of the Kernel-Machine Library                                     *
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

#ifndef KML_ONLINE_GP_HPP
#define KML_ONLINE_GP_HPP

#include <boost/numeric/bindings/traits/std_vector.hpp>
#include <boost/numeric/bindings/traits/ublas_matrix.hpp>
#include <boost/numeric/bindings/traits/ublas_symmetric.hpp>
#include <boost/numeric/bindings/traits/ublas_vector.hpp>
#include <boost/mpl/equal_to.hpp>

#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/tracking.hpp>
#include <boost/serialization/vector.hpp>

#include <boost/type_traits/is_float.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/utility/enable_if.hpp>

#include <kml/matrix_view.hpp>
#include <kml/symmetric_view.hpp>

#include <kml/regression.hpp>
#include <kml/classification.hpp>
#include <kml/ranking.hpp>

#include <kml/kernel_machine.hpp>

#include <kml/moments.hpp>

#include <boost/numeric/ublas/io.hpp>



namespace atlas = boost::numeric::bindings::atlas;
namespace mpl = boost::mpl;

namespace kml {


/*

Description


References
Csató and Opper, 2002, Sparse Online Gaussian Processes


NOTE: the implementation is not finished yet, and may be subject to numerous changes.


-----------------------------
symbol translation guide
-----------------------------
here       paper(s), website     matlab code

weight     alpha                 net.w
k_tt       k^star_{t+1}          kk
k_t        k_t                   kX
delta_t    gamma_{t+1}, q^star   gamma
a_t        \hat{e}_{t+1}         hatE
s_t        s_{t+1}               tVector
R          K^inv_t; Q_t          net.KBinv
C          C                     net.C
.D1()      q^{t+1}               K1
.D2()      r^{t+1}               K2


*/


class gaussian_noise { 
public:
	void operator()( double y, double f_y, double cov_t ) {

		m_d2 = -1.0 / cov_t;
		m_d1 = (y - f_y) / cov_t;
	}
	double D1() {
		return m_d1;
	}
	double D2() {
		return m_d2;
	}
private:
	double m_d1;
	double m_d2;
};


/*

On-line Gaussian Processes.

*/


template< typename Problem, typename Kernel, typename PropertyMap, class Enable = void>
class online_gp: public kernel_machine<PropertyMap,Problem,Kernel> {};

//
//
// REGRESSION ALGORITHM
//
//

template< typename Problem, typename Kernel, typename PropertyMap >
class online_gp< Problem, Kernel, PropertyMap, typename boost::enable_if< is_regression<Problem> >::type>:
    public kernel_machine< Problem, Kernel, PropertyMap > {
public:
    typedef kernel_machine< Problem, Kernel, PropertyMap > base_type;
    typedef typename base_type::kernel_type kernel_type;
    typedef typename base_type::result_type result_type;
    typedef typename Problem::input_type input_type;
    typedef typename Problem::output_type output_type;

    typedef double scalar_type;
    typedef boost::tuple<double,double> tuple_type;

    typedef symmetric_view< ublas::matrix<double> > symmetric_type;
    typedef ublas::matrix<double> matrix_type;
    typedef ublas::vector<double> vector_type;

    //friend class boost::serialization::access;

    typedef typename boost::property_traits<PropertyMap>::key_type key_type;
    typedef typename boost::property_traits<PropertyMap>::value_type object_type;

    online_gp( scalar_type const tol,
               typename boost::call_traits<kernel_type>::param_type k,
               typename boost::call_traits<PropertyMap>::param_type map ):
    base_type(k,map), tolerance(tol) {}


    two_moments operator()( input_type const &x ) {

        vector_type temp_K( basis_key.size() );
        fill_kernel( x, basis_key.begin(), basis_key.end(), temp_K.begin() );

        ublas::matrix_range< ublas::matrix<double> > C_range( C.view() );
        ublas::symmetric_adaptor< ublas::matrix_range< ublas::matrix<double> > > C_view( C_range );
        vector_type temp_vector( basis_key.size() );
	atlas::symv( C_view, temp_K, temp_vector );

	// noise is model dependent
	return two_moments( atlas::dot( weight, temp_K ), atlas::dot( temp_vector, temp_K ) + kernel( x, x ) + variance_estimate );
    }


    template< typename KeyIterator >
    void learn( KeyIterator begin, KeyIterator end ) {
	KeyIterator iter(begin);
	while( iter != end ) {
		increment( *iter++ );
	}
    }

    void increment( key_type const &key ) {

	variance_estimate = sqrt(0.2);

        // calculate the base_type::kernel function on (x_t,x_t), needed later on
        scalar_type k_tt = kernel( key, key );

         // fill vector k_t
	k_t.resize( basis_key.size() );
        fill_kernel( key, basis_key.begin(), basis_key.end(), k_t.begin() );

        // a_t <- R %*% k_t
	// this version of symv overwrites everything in a_t
	a_t.resize( basis_key.size() );
        ublas::matrix_range< ublas::matrix<double> > R_range( R.view() );
        ublas::symmetric_adaptor< ublas::matrix_range< ublas::matrix<double> > > R_view( R_range );
        atlas::symv( R_view, k_t, a_t );

        // s_t <- C %*% k_t  (will be further updated according to code path)
	// this version of symv overwrites everything in s_t
        ublas::matrix_range< ublas::matrix<double> > C_range( C.view() );
        ublas::symmetric_adaptor< ublas::matrix_range< ublas::matrix<double> > > C_view( C_range );
        atlas::symv( C_view, k_t, s_t );

        // update the first and second order derivatives of the loglikelihood
	// sigma_0: ??
	// Gaussian case: see above Equation 31, page 9
	// sigma_0 = prior variance estimate (?) ...
	scalar_type cov_t = variance_estimate + k_tt + atlas::dot( k_t, s_t );

	// TODO -- make this noise-model dependent 
	likelihood( output(key), atlas::dot( k_t, weight ), cov_t );

	// delta_t <- k_tt - k_t %*% K %*% k_t == k_tt - k_t %*% a_t (see Equation 23)
        scalar_type delta_t = k_tt - atlas::dot( k_t, a_t );

        // Perform Approximate Linear Dependency (ALD) test
        // If the ALD is larger than some previously set tolerance, add the input to 
        // the dictionary (or basis vector set)
        if ( delta_t > tolerance ) {

                // add x_t to support vector set, adjust all needed variables
                std::size_t old_size( basis_key.size() );

                // update matrix C; Equation 9, middle equation
		atlas::syr( likelihood.D2(), s_t, C_view );
                C.grow_row_column();
                ublas::matrix_vector_slice< ublas::matrix<double> > C_row_part( C.shrinked_row(old_size) );
		atlas::scal( likelihood.D2(), s_t );
		atlas::copy( s_t, C_row_part );
		C.matrix( old_size, old_size ) = likelihood.D2();

                 // update inverse matrix R; Appendix D
                scalar_type factor = static_cast<scalar_type>(1) / delta_t;
                atlas::syr( factor, a_t, R_view );
                R.grow_row_column();
                ublas::matrix_vector_slice< ublas::matrix<double> > R_row_part( R.shrinked_row(old_size) );
                atlas::scal( -factor, a_t );
		atlas::copy( a_t, R_row_part );
                R.matrix( old_size, old_size ) = factor;

		// further estimate s_t; Equation 9, bottom equation
		s_t.push_back( 1.0 );

                // adjust the weight vector; Equation 9, top equation
		weight.push_back( 0.0 );
		atlas::axpy( likelihood.D1(), s_t, weight );

		// add this key to the basis vector set
		basis_key.push_back( key );

	} else {
		// further estimate s_t; Equations 11, 16
		atlas::xpy( a_t, s_t );

		// compute the scalar eta_{t+1}
		double eta = 1.0 / ( 1.0 + delta_t * likelihood.D2() );

                // adjust weight vector; Equation 88, thesis, top equation
                atlas::axpy( likelihood.D1() * eta, s_t, weight );

                // update permutation matrix (equation 14)
                atlas::syr( likelihood.D2() * eta, s_t, C_view );

	}


    } // increment method

    // these are the parameters
    gaussian_noise likelihood;

    scalar_type tolerance;   // the ALD tolerance parameter
    scalar_type variance_estimate;    // the variance estimate

    symmetric_type R;	// this is the inverse of the {gram/kernel} matrix
    symmetric_type C;	// this is the C matrix

    std::vector< scalar_type > k_t;    // a vector with evaluated kernel values for each basis vector
    std::vector< scalar_type > a_t;    // a_t <- K %*% k_t
    std::vector< scalar_type > s_t;    // depends on reestimate or add_to_basis; s_t <- C %*% k_t + "depends"

    std::vector< key_type > basis_key;   // a vector containing basis vector keys
    std::vector< scalar_type > weight;   // weights associated with the basis vectors

};



} // namespace kml






#endif

