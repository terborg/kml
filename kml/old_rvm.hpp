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

#ifndef OLD_RVM_HPP
#define OLD_RVM_HPP

#include <boost/call_traits.hpp>
#include <boost/ref.hpp>
#include <boost/range/begin.hpp>
#include <boost/range/end.hpp>
#include <boost/range/size.hpp>
#include <ext/numeric>

#include <kml/kernel_machine.hpp>
#include <kml/design_matrix.hpp>

// TODO fix old and kludgy algorithms from kernels.h
#include "kernels.h"


// include AFTER enough traits have been included, of course... (?)
#include <boost/numeric/bindings/atlas/cblas.hpp>
#include <boost/numeric/bindings/atlas/clapack.hpp>
#include <boost/numeric/bindings/lapack/ppsv.hpp>


namespace atlas = boost::numeric::bindings::atlas;
namespace lapack = boost::numeric::bindings::lapack;



/*! Classic RVM algorithm */

namespace kml {



template<typename Problem,template<typename,int> class Kernel,class Enable = void>
class old_rvm: public kernel_machine<Problem,Kernel> {};


template<typename Problem,template<typename,int> class Kernel>
class old_rvm<Problem,Kernel,typename boost::enable_if< is_regression<Problem> >::type>:
         public kernel_machine<Problem,Kernel> {
public:
    typedef kernel_machine<Problem,Kernel> base_type;
    typedef typename base_type::kernel_type kernel_type;
    typedef typename base_type::result_type result_type;
    typedef typename Problem::input_type input_type;
    typedef typename Problem::output_type output_type;
    typedef double scalar_type;
    typedef ublas::symmetric_matrix<double> symmetric_type;
    typedef ublas::matrix<double> matrix_type;
    typedef ublas::vector<double> vector_type;

    /*! \param k the kernel construction parameter */
    old_rvm( typename boost::call_traits<kernel_type>::param_type k ): base_type(k) {}

    template<class IRange, class ORange>
    void learn( IRange const &input, ORange const &output ) {
 
        // create the kernel matrix (symmetric part)
        // initialise & compute HtH, Hty
        matrix_type H;
        design_matrix( input, input, base_type::kernel, 1.0, H );
 
        vector_type Hty( input.size()+1 );
        atlas::gemv( CblasTrans, 1.0, H, output, 0.0, Hty );    
 
        symmetric_type HtH( H.size2(), H.size2() );
        inner_prod( H, HtH );
 
        // explanation of variables, following Tipping (1999)
        // theta = 1 / alpha
 
        std::vector<unsigned int> indices( HtH.size1() );
        // fill indices with 0,1,2,3,...,N-1. This is not in standard C++, but an SGI extension.
        iota( indices.begin(), indices.end(), 0 );
 
        vector_type weight_vector( HtH.size1() );
        vector_type theta( HtH.size2() );
 
        // theta is 1/alpha, as in Ralf Herbrich's R implementation
        // since near-zero values are better defined than "very large" values
        atlas::set
            ( 1.0, theta );
        atlas::set
            ( 1.0, weight_vector );
 
        variance_estimate = 0.1;
 
        scalar_type max_diff = 1.0;
 
        // RVM's main loop
        while( max_diff > 1e-3 ) {
 
            // Prepare the not-yet-inverted part of matrix sigma, and
            // the non-multiplied part of vector mu
            // NOTE not this anymore, changed to more efficient formulation
            //      see above
            // sigma_inv <- (HtH*(1/var)+A)  (part of eq. 5)
            // mu        <- Hty              (part of eq. 6)
            ublas::symmetric_matrix<double, ublas::lower, ublas::column_major> sigma_inverted( indices.size(), indices.size() );
            vector_type mu( indices.size() );
            for( unsigned int i=0; i<indices.size(); ++i ) {
                unsigned int index_sv = indices[i];
                mu( i ) = Hty( index_sv );
                for( unsigned int col=0; col <= i; ++col ) {
                    sigma_inverted(i,col) = HtH( index_sv, indices[col] );
                }
                sigma_inverted(i,i) += variance_estimate / theta(index_sv);
            }
 
            // Invert matrix to form matrix sigma and compute vector mu
            // sigma <- (HtH*(1/var)+A)^-1 (equation 5, Tipping, 1999)
            // mu    <- (1/var)*sigma*Hty  (equation 6, Tipping, 1999)
            // TODO symmetric adaptor can go if upper or lower is known (other call to symv)
            // NOTE recip. variance is probably not most efficient like this.
 
            ublas::matrix<double, ublas::column_major> sigma( indices.size(), indices.size() );
            // temporary workaround for boost non-compileness stuff.
            //ublas::matrix<double, ublas::column_major> sigma( ublas::identity_matrix<double>( indices.size() ) );
            sigma.clear();
            for( unsigned int i=0; i<indices.size(); ++i ) {
                sigma(i,i)=1.0;
            }
 
 
            ublas::symmetric_adaptor<ublas::matrix<double, ublas::column_major> > sigma_symm( sigma );
            lapack::ppsv( sigma_inverted, sigma );
 
            // FIXME weird behaviour of atlas when doing this multiply with 2 mu's in there
            // TODO just multiply with the right part of Hty; do not copy this to mu first
            // TODO atlas-ify
            //atlas::gemv( sigma, mu, mu );
            mu = ublas::prod( sigma, mu );
 
            // update theta vector
            // compute sum_i gamma_i
            scalar_type gamma_summed = static_cast<scalar_type>(0);
            max_diff = static_cast<scalar_type>(0);
            for( unsigned int i=0; i<indices.size(); ++i ) {
                unsigned int index_sv = indices[i];
                scalar_type gamma_i = 1.0 - ((variance_estimate * sigma(i,i)) / theta(index_sv));
                scalar_type theta_old = theta( index_sv );
                theta( index_sv ) = (mu(i) * mu(i)) / gamma_i;
                gamma_summed += gamma_i;
                weight_vector( index_sv ) = mu(i);
                scalar_type diff = std::fabs( std::log(theta_old) - log(theta(index_sv)) );
                if ( diff > max_diff )
                    max_diff=diff;
            }
 
            // std::cout << "vector gamma: " << gamma << std::endl;
            // std::cout << "vector theta: " << theta << std::endl;
            variance_estimate = residual_sum_squares( H, weight_vector, indices, output ) / (static_cast<scalar_type>(output.size()) - gamma_summed );
 
            // prune support vectors on the basis of the theta vector
            prune_indices( indices, theta );
 
        }
 
 
	if (debug ) {
        	std::cout << "Relevance Vector Machine (Tipping, 1999)" << std::endl;
        	std::cout << "support vectors: ";
        	std::copy( indices.begin(), indices.end(), std::ostream_iterator<scalar_type>(std::cout, " ") );
        	std::cout << std::endl;
        	std::cout << "variance estimate: " << variance_estimate << std::endl;
        	std::cout << std::endl;
	}
 
        //         std::cout << "weights: " << std::endl;
        //         for( unsigned int i=0; i<indices.size(); ++i ) {
        //           std::cout << weight_vector[i] << " ";
        // 	}
        //         std::cout << std::endl;
 

       if ( indices.front() == 0 ) {
           base_type::bias = weight_vector(0);
           indices.erase( indices.begin() );
       } else {
           base_type::bias = 0.0;
       }

       base_type::weight.resize( indices.size() );
       base_type::support_vector.resize( indices.size() );
       for( unsigned int i=0; i<indices.size(); ++i ) {
   	   base_type::weight[i] = weight_vector( indices[i] );
	   base_type::support_vector[i] = input[ indices[i]-1 ];
       }


        // 	int qq;
        // 	std::cin >> qq;
 
    }
 
 
public:
    static const bool debug = false; 

   // locally stored
    scalar_type variance_estimate;
 
};


} // namespace kml


#endif

