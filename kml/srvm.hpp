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

#ifndef SRVM_HPP
#define SRVM_HPP

#include <ext/numeric>

#include <boost/numeric/bindings/lapack/ppsv.hpp>
// #include "traits.hpp"
//
#include "kernels.h"

#include <design_matrix.hpp>

//
#include <map>


#include <boost/numeric/bindings/traits/ublas_matrix.hpp>



/*namespace ublas = boost::numeric::ublas;
namespace atlas = boost::numeric::bindings::atlas;*/
namespace lapack = boost::numeric::bindings::lapack;



namespace kml {

/*!
\brief Smooth Relevance Vector Machine
\param I the input type
\param O the output type
\param K the kernel type

This class implements a reformulation of the classic Relevance Vector Machine
to use a roughness penalty as well. This results in a function which will have
derivatives that "behave well", i.e. are continuous up to some order.


\section bibliography References
-# Rutger ter Borg and Leon Rothkrantz. Smooth Bayesian Kernel Machines. 

*/

template<typename I, typename O, template<typename,int> class K >
class srvm: public determinate<I,O,K> {
public:
    typedef determinate<I,O,K> base_type;
    typedef typename base_type::kernel_type kernel_type;
    typedef typename base_type::result_type result_type;
    typedef I input_type;
    typedef O output_type;
    typedef double scalar_type;
    typedef ublas::symmetric_matrix<double> symmetric_type;
    typedef ublas::matrix<double> matrix_type;
    typedef ublas::vector<double> vector_type;


    typedef std::vector<unsigned int> index_type;


    srvm( typename boost::call_traits<kernel_type>::param_type k ): base_type(k) {}


    template<class IRange, class ORange>
    void learn( IRange const &input, ORange const &output ) {
        learn( input, input, output );
    }


    template<class IRange, class ORange>
    void learn( IRange const &source, IRange const &target, ORange const &output ) {


        // create the kernel matrix (symmetric part)
        // initialise & compute HtH, Hty
        matrix_type H;
        kml::design_matrix( source, target, base_type::kernel, 1.0, H );

        // TODO ublas::trans functionality has changed since boost 1.32
	// TODO use a less heavy call?
        vector_type Hty( source.size()+1 );
        atlas::gemv( CblasTrans, 1.0, H, output, 0.0, Hty );    
	 	
	// output == std::vector, is unknown to ublas::prod.
	//vector_type Hty( ublas::prod( ublas::trans(H), output) );

        symmetric_type HtH( H.size2(), H.size2() );
        inner_prod( H, HtH );

        // TEST TEST TEST
        matrix_type L;
        typename base_type::template kernel_derivative<6>::type kernel2( base_type::kernel.get_parameter() );
        kml::design_matrix( source, target, kernel2, 0.0, L );
        symmetric_type LtL( H.size2() ); //, H.size2() );
        inner_prod( L, LtL );

        // explanation of variables, following Tipping (1999)
        // theta = 1 / alpha

        std::vector<unsigned int> indices( HtH.size1() );
        // fill indices with 0,1,2,3,...,N-1. This is not in standard C++, but an SGI extension.
        iota( indices.begin(), indices.end(), 0 );

        vector_type weight_vector( HtH.size1() );
        vector_type theta( HtH.size2() );

        // theta is 1/alpha, as in Ralf Herbrich's R implementation
        // since near-zero values are better defined than "very large" values
        // TODO atlas/standard routine
        /*	atlas::set( 1.0, theta );
                atlas::set( 1.0, weight_vector );*/
        std::fill( weight_vector.begin(), weight_vector.end(), 1.0 );
        std::fill( theta.begin(), theta.end(), 1.0 );


        variance = 0.1;
        roughness_penalty = 0.1;


        scalar_type max_diff = 1.0;

        // main loop

	
        // Change stopping criterion?

        while( max_diff > 1e-3 ) {

            /*		std::cout << std::endl;*/

            // Prepare the not-yet-inverted part of matrix sigma, and
            // the non-multiplied part of vector mu
            // sigma_inv <- (HtH*(1/var)+A)  (part of eq. 5)
            // mu        <- Hty              (part of eq. 6)
            scalar_type reciprocal_variance = static_cast<scalar_type>(1.0) / variance;
            ublas::symmetric_matrix<double, ublas::lower, ublas::column_major> sigma_inverted( indices.size(), indices.size() );
            ublas::symmetric_matrix<double, ublas::lower, ublas::column_major> S_inverted( indices.size(), indices.size() );
            vector_type mu( indices.size() );

            /*
            	    std::cout << "theta's in use: ";	    */

            for( unsigned int i=0; i<indices.size(); ++i ) {
                unsigned int index_sv = indices[i];
                mu( i ) = Hty( index_sv );
                for( unsigned int col=0; col <= i; ++col ) {
                    scalar_type S_inv_ij = roughness_penalty * LtL( index_sv, indices[col] );
                    S_inverted(i,col) = S_inv_ij;
                    sigma_inverted(i,col) = reciprocal_variance * HtH( index_sv, indices[col] ) + S_inv_ij;
                }
                sigma_inverted(i,i) += static_cast<scalar_type>(1.0) / theta(index_sv);
                /*		std::cout << theta(index_sv) << " ";*/
            }
            /*		std::cout << std::endl;*/

            // Invert matrix to form matrix sigma and compute vector mu
            // TODO symmetric adaptor can go if upper or lower is known (other call to symv)
            // NOTE recip. variance is probably not most efficient like this.
            ublas::matrix<double, ublas::column_major> sigma( indices.size(), indices.size() );
            ublas::matrix<double, ublas::column_major> S( indices.size(), indices.size() );

            // temporary workaround for boost non-compileness stuff.
            //ublas::matrix<double, ublas::column_major> sigma( ublas::identity_matrix<double>( indices.size() ) );
            sigma.clear();
            S.clear();
            for( unsigned int i=0; i<indices.size(); ++i ) {
                sigma(i,i)=1.0;
                S(i,i)=1.0;
            }

            // FIXME what if matrix sigma or S is empty?
            ublas::symmetric_adaptor<ublas::matrix<double, ublas::column_major> > sigma_symm( sigma );
            lapack::ppsv( sigma_inverted, sigma );

            // I think this is INCORRECT . . . . (check reference ...... )
            //atlas::symv( reciprocal_variance, sigma_symm, mu, 0.0, mu );
            // TODO FIXME NOTE check this!!
            // IMPORTANT ! ! ! ! WAS A CAUSE OF A BUG! ! ! ! ! ! ! ! !
            mu = reciprocal_variance * ublas::prod( sigma_symm, mu );

            /*	   std::cout << "vector mu:    " << mu << std::endl;*/



            // alright, S_inverted contains a subpart of roughness_penalty*LtL at this point
            // mu contains mu
            // we need: t(mu) %*% LtL %*% mu
            // so... compute this...
            vector_type temp( indices.size() );
            scalar_type muLtLmu = 0;
            if (roughness_penalty > 0.0) {
                atlas::spmv( 1.0/roughness_penalty, S_inverted, mu, 0.0, temp );
                muLtLmu = atlas::dot( mu, temp );
            }
            // 	    std::cout << "muLtLmu is " << muLtLmu << std::endl;

            // Sigma is also computed at this point, so we can determine the trace of Sigma*roughness_penalty*LtL
            scalar_type trace_Sigma_LtL = static_cast<scalar_type>(0);
            for( unsigned int i=0; i<indices.size(); ++i ) {
                trace_Sigma_LtL += ublas::inner_prod( ublas::row(sigma,i), ublas::column(S_inverted,i) );//sigma_row, LtL_col );
            }

            // complete S_inv by adding A to it
            for( unsigned int i=0; i<indices.size(); ++i ) {
                S_inverted(i,i) += static_cast<scalar_type>(1.0) / theta(indices[i]);
            }
            lapack::ppsv( S_inverted, S );

            // update theta vector
            // compute sum_i gamma_i

            scalar_type trace_Sigma_A = static_cast<scalar_type>(0);
            scalar_type trace_S_A = static_cast<scalar_type>(0);

            scalar_type gamma_summed = static_cast<scalar_type>(0);
            max_diff = static_cast<scalar_type>(0);
            for( unsigned int i=0; i<indices.size(); ++i ) {
                unsigned int index_sv = indices[i];
                scalar_type theta_old = theta( index_sv );

                trace_Sigma_A += sigma(i,i) / theta_old;
                trace_S_A += S(i,i) / theta_old;

                //std::cout << (1.0/theta( index_sv )) * S(i,i) << std::endl;
                
		scalar_type gamma_i = 1.0 - (sigma(i,i) / theta(index_sv));
                theta( index_sv ) = std::max<scalar_type>( 0.0, (mu(i) * mu(i)) / ((S(i,i)-sigma(i,i))/theta_old) );
                gamma_summed += gamma_i;
                weight_vector( index_sv ) = mu(i);
                scalar_type diff = std::fabs( std::log(theta_old) - log(theta(index_sv)) );
                if ( diff > max_diff )
                    max_diff=diff;
            }


            variance = residual_sum_squares( H, weight_vector, indices, output ) /
                       (static_cast<scalar_type>(output.size()) - static_cast<scalar_type>(indices.size()) +
                        trace_Sigma_LtL + trace_Sigma_A );

            if (muLtLmu > 0.0) {
                roughness_penalty = (static_cast<scalar_type>(indices.size()) - trace_S_A - trace_Sigma_LtL) / muLtLmu;
            } else
                roughness_penalty=0.0;

	    // prune support vectors on the basis of the theta vector
            prune_indices( indices, theta );

            for( unsigned int i=0; i<indices.size(); ++i ) {
                if (theta( indices[i] ) > (1.0/std::numeric_limits<double>::epsilon())) {
                    theta(indices[i]) = std::numeric_limits<double>::infinity();
                }
            }

        }


        std::cout << "Smooth Relevance Vector Machine (Ter Borg, 2005)" << std::endl;
        std::cout << "Support vectors:               " << indices.size() << std::endl;
        std::cout << "Variance estimate:             " << variance << std::endl;
        std::cout << "Smoothness parameter estimate: " << roughness_penalty << std::endl;

        if ( indices.front() == 0 ) {
            base_type::bias = weight_vector(0);
            indices.erase( indices.begin() );
        } else {
            base_type::bias = 0.0;
        }

        base_type::weight.resize( indices.size() );
        base_type::support_vector.resize( indices.size() );
        for( int i=0; i<indices.size(); ++i ) {
            base_type::weight[i] = weight_vector( indices[i] );
            base_type::support_vector[i] = source[ indices[i]-1 ];
        }
    
    }


    
    scalar_type variance;
    scalar_type roughness_penalty;

};






} //namespace kml












/*
 
    template<typename MatT, typename VecT, class KF>
    void train2( MatT const &X, VecT const &y, KF const &kernel = KF() ) {
 
        // create the kernel matrix (symmetric part)
        // initialise & compute HtH, Hty
        matrix_type H;
        kml::design_matrix( X, X, kernel, 1.0, H );
 
        vector_type Hty( ublas::prod( ublas::trans(H), y ) );
        //atlas::gemv(ublas::trans(H), y, Hty);
 
        symmetric_type HtH( H.size2(), H.size2() );
        inner_prod( H, HtH );
 
 
        std::map<unsigned int, matrix_type> L;
        std::map<unsigned int, double> lambda;
        std::vector<unsigned int> orders;
 
        orders.push_back( 4 );
        kml::gaussian<ublas::vector<double>,4 > kernelD4( kernel.param );
        kml::design_matrix( X, X, kernelD4, 0.0, L[4] );
        lambda[4] = 0.5;
 
        orders.push_back( 6 );
        kml::gaussian<ublas::vector<double>,6 > kernelD6( kernel.param );
        kml::design_matrix( X, X, kernelD6, 0.0, L[6] );
        lambda[6] = 0.5;
 
        orders.push_back( 8 );
        kml::gaussian<ublas::vector<double>,8 > kernelD8( kernel.param );
        kml::design_matrix( X, X, kernelD8, 0.0, L[8] );
        lambda[8] = 0.5;
 
        
        //	orders.push_back( 10 );
        //        Gaussian<double,10> kernelD10( sigma(1.6) );
        //        kml::design_matrix( X, X, kernelD10, L[10] );
      //  	lambda[10] = 0.5;
         
        	orders.push_back( 12 );
                Gaussian<double,12> kernelD12( sigma(1.6) );
                kml::design_matrix( X, X, kernelD12, L[12] );
        	lambda[12] = 0.1;
        
 
        // define the matrix we will use as temporary
        matrix_type L_sum( L[4].size1(), L[4].size2() );
 
        symmetric_type LtL( H.size2(), H.size2() );
 
        matrix_type LitL( H.size2(), H.size2() );
 
 
 
        //HtH += 0.1 * LtL;
        // TEST TEST TEST
 
 
        // explanation of variables, following Tipping (1999)
        // theta = 1 / alpha
 
        std::vector<unsigned int> indices( HtH.size1() );
        // fill indices with 0,1,2,3,...,N-1. This is not in standard C++, but an SGI extension.
        iota( indices.begin(), indices.end(), 0 );
 
        vector_type weight_vector( HtH.size1() );
        vector_type theta( HtH.size2() );
 
        // theta is 1/alpha, as in Ralf Herbrich's R implementation
        // since near-zero values are better defined than "very large" values
        //atlas::set( 1.0, theta );
        //atlas::set( 1.0, weight_vector );
        std::fill( theta.begin(), theta.end(), 1.0 );
        std::fill( weight_vector.begin(), weight_vector.end(), 1.0 );
 
        variance_estimate = 0.1;
        scalar_type max_diff = 1.0;
 
        int iters=0;
 
 
        // RVM's main loop
        while( max_diff > 1e-3 ) {
 
 
            std::cout << "iteration " << iters++ << std::endl;
 
            // re-compute LtL, on the basis of all lambda's and L's
            // TODO: automatic loop unrollment for this expression
            noalias(L_sum) = lambda[4]*L[4] + lambda[6]*L[6] + lambda[8]*L[8]; // + lambda[10]*L[10] + lambda[12]*L[12];
            //	   noalias(L_sum) = lambda[12]*L[12];
            std::cout << "..." << std::endl;
 
            inner_prod( L_sum, LtL );
 
 
            // Prepare the not-yet-inverted part of matrix sigma, and
            // the non-multiplied part of vector mu
            // sigma_inv <- (HtH*(1/var)+A)  (part of eq. 5)
            // mu        <- Hty              (part of eq. 6)
            scalar_type reciprocal_variance = static_cast<scalar_type>(1.0) / variance_estimate;
            ublas::symmetric_matrix<double, ublas::lower, ublas::column_major> sigma_inverted( indices.size(), indices.size() );
            ublas::symmetric_matrix<double, ublas::lower, ublas::column_major> S_inverted( indices.size(), indices.size() );
            vector_type mu( indices.size() );
 
 
            for( unsigned int i=0; i<indices.size(); ++i ) {
                unsigned int index_sv = indices[i];
                mu( i ) = Hty( index_sv );
                for( unsigned int col=0; col <= i; ++col ) {
                    scalar_type S_inv_ij = LtL( index_sv, indices[col] );
                    S_inverted(i,col) = S_inv_ij;
                    sigma_inverted(i,col) = reciprocal_variance * HtH( index_sv, indices[col] ) + S_inv_ij;
                }
                sigma_inverted(i,i) += static_cast<scalar_type>(1.0) / theta(index_sv);
            }
 
            // Invert matrix to form matrix sigma and compute vector mu
            // sigma <- (HtH*(1/var)+A)^-1 (equation 5, Tipping, 1999)
            // mu    <- (1/var)*sigma*Hty  (equation 6, Tipping, 1999)
            // TODO symmetric adaptor can go if upper or lower is known (other call to symv)
            // NOTE recip. variance is probably not most efficient like this.
            ublas::matrix<double, ublas::column_major> sigma( indices.size(), indices.size() );
            ublas::matrix<double, ublas::column_major> S( indices.size(), indices.size() );
 
            // temporary workaround for boost non-compileness stuff.
            //ublas::matrix<double, ublas::column_major> sigma( ublas::identity_matrix<double>( indices.size() ) );
            sigma.clear();
            S.clear();
            for( unsigned int i=0; i<indices.size(); ++i ) {
                sigma(i,i)=1.0;
                S(i,i)=1.0;
            }
 
            ublas::symmetric_adaptor<ublas::matrix<double, ublas::column_major> > sigma_symm( sigma );
            lapack::ppsv( sigma_inverted, sigma );
            // NOTE TODO BUG ! ! ! ! ! ! ! ! ! ! ! ! ! !
            atlas::symv( reciprocal_variance, sigma_symm, mu, 0.0, mu );
 
            // Sigma is also computed at this point, so we can determine the trace of Sigma*LtL
            scalar_type trace_Sigma_LtL = static_cast<scalar_type>(0);
            for( unsigned int i=0; i<indices.size(); ++i ) {
                ublas::matrix_row<ublas::matrix<double, ublas::column_major> > sigma_row( sigma, i );
                ublas::matrix_column<ublas::symmetric_matrix<double, ublas::lower, ublas::column_major> > LtL_col( S_inverted, i );
                trace_Sigma_LtL += ublas::inner_prod( sigma_row, LtL_col );
            }
 
            //std::cout << "trace Sigma LtL is " << trace_Sigma_LtL << std::endl;
 
            // complete S_inv by adding A to it
            for( unsigned int i=0; i<indices.size(); ++i ) {
                S_inverted(i,i) += static_cast<scalar_type>(1.0) / theta(indices[i]);
            }
            lapack::ppsv( S_inverted, S );
 
            // update theta vector
            // compute sum_i gamma_i
 
            //scalar_type trace_SA = static_cast<scalar_type>(0);
 
            scalar_type trace_Sigma_A = static_cast<scalar_type>(0);
            scalar_type trace_S_A = static_cast<scalar_type>(0);
 
            scalar_type gamma_summed = static_cast<scalar_type>(0);
            max_diff = static_cast<scalar_type>(0);
            for( unsigned int i=0; i<indices.size(); ++i ) {
                unsigned int index_sv = indices[i];
                scalar_type theta_old = theta( index_sv );
 
                trace_Sigma_A += sigma(i,i) / theta_old;
                trace_S_A += S(i,i) / theta_old;
 
                //std::cout << (1.0/theta( index_sv )) * S(i,i) << std::endl;
 
 
                scalar_type gamma_i = 1.0 - (sigma(i,i) / theta(index_sv));
 
 
                theta( index_sv ) = (mu(i) * mu(i)) / ((S(i,i)-sigma(i,i))/theta_old);
 
 
                gamma_summed += gamma_i;
 
 
                weight_vector( index_sv ) = mu(i);
                scalar_type diff = std::fabs( std::log(theta_old) - log(theta(index_sv)) );
                if ( diff > max_diff )
                    max_diff=diff;
            }
 
            std::cout << "trace Sigma_A:   " << trace_Sigma_A << std::endl;
            std::cout << "trace S_A:       " << trace_S_A << std::endl;
            std::cout << "trace Sigma_LtL: " << trace_Sigma_LtL << std::endl;
 
            std::cout << "gamma summed is " << gamma_summed << std::endl;
            std::cout << "alternative is  " << -static_cast<scalar_type>(indices.size()) + trace_Sigma_A + trace_Sigma_LtL << std::endl;
 
            std::cout << "N-Tr(Sigma*sigma^-2*HtH) = " << static_cast<scalar_type>(y.size())-static_cast<scalar_type>(indices.size()) + trace_Sigma_A + trace_Sigma_LtL << std::endl;
 
            // std::cout << "vector gamma: " << gamma << std::endl;
            std::cout << "vector mu: " << mu << std::endl;
            variance_estimate = residual_sum_squares( H, weight_vector, indices, y ) /
                                (static_cast<scalar_type>(y.size()) - static_cast<scalar_type>(indices.size()) + trace_Sigma_LtL + trace_Sigma_A );
 
            std::cout << "current parameters lambda: ";
            for( unsigned int i=0; i<orders.size(); ++i ) {
                std::cout << lambda[orders[i]] << " ";
            }
            std::cout << std::endl;
 
 
            // alright, S_inverted contains a subpart of lambda*LtL at this point
            // mu contains mu
            // we need: t(mu) %*% LtL %*% mu
            // so... compute this...
            vector_type temp( indices.size() );
            scalar_type muLtLmu = 0;
            //	    if (lambda > 0.0) {
            //  	      atlas::spmv( 1.0/lambda, S_inverted, mu, 0.0, temp );
           // 	      muLtLmu = atlas::dot( mu, temp );
          //  	    }
            //std::cout << "muLtLmu is " << muLtLmu << std::endl;
 
 
            for( unsigned int i=0; i<orders.size(); ++i ) {
                unsigned int N = orders[i];
                std::cout << "order under consideration: " << N << std::endl;
 
                vector_type tempL_i( L_sum.size1() );
                vector_type tempL( L_sum.size1() );
                tempL.clear();
                tempL_i.clear();
                std::cout << "..." << std::endl;
 
                // compute expensive inner product
                inner_prod( L[N], L_sum, LitL );
 
                scalar_type trace_S_Sigma_LitL = 0;
                for( unsigned j=0; j<indices.size(); ++j ) {
                    //std::cout << "index " << indices[j] << std::endl;
                    // in L, all lambda's are present. In L[N], no lambda is present. So, put that in.
 
                    scalar_type id_in_prod = 0;
                    for( unsigned q=0; q<indices.size(); ++q ) {
                        id_in_prod += (S(j,q)-sigma(j,q)) * LitL( indices[q], indices[j] );
                    }
                    trace_S_Sigma_LitL += id_in_prod;
 
                    noalias(tempL) += column(L_sum, indices[j]) * mu(j);  // <-- CAN BE REMOVED FROM THIS LOOP!!!!
                    noalias(tempL_i) += column(L[N], indices[j]) * mu(j);
                }
 
                std::cout << "trace_S_Sigma_LiTL = " << trace_S_Sigma_LitL << std::endl;
 
                std::cout << "mutLitLmu = " << atlas::dot( tempL, tempL_i ) << std::endl;
 
                std::cout << "potential new lambda: " << sqrt( lambda[N]*lambda[N] * trace_S_Sigma_LitL / atlas::dot( tempL, tempL_i ) ) << std::endl;
 
 
                lambda[N] = sqrt( lambda[N]*lambda[N] * trace_S_Sigma_LitL / atlas::dot( tempL, tempL_i ) );
                //lambda[N] = lambda[N] * trace_S_Sigma_LitL / atlas::dot( tempL, tempL_i );
 
            }
 
            std::cout << "current parameters lambda: ";
            for( unsigned int i=0; i<orders.size(); ++i ) {
                std::cout << lambda[orders[i]] << " ";
            }
            std::cout << std::endl;
 
 
 
 
 
            // 	    std::cout << "current lambda: " << lambda << std::endl;
 
            //	    if (muLtLmu > 0.0) {
           // 	      lambda = (static_cast<scalar_type>(indices.size()) - trace_S_A - trace_Sigma_LtL) / muLtLmu;
          //  	    } else lambda=0.0;
            // 	    std::cout << "current lambda: " << lambda << std::endl;
 
 
 
 
            // prune support vectors on the basis of the theta vector
            // what to do? 1) prune if below singularity limit of matrix
            // 2) ehhh......
            prune_indices( indices, theta );
 
            int qq;
            std::cin >> qq;
 
        }
 
 
        std::cout << "Smooth Relevance Vector Machine (Ter Borg, 2004)" << std::endl;
        std::cout << "support vectors: ";
        std::copy( indices.begin(), indices.end(), std::ostream_iterator<scalar_type>(std::cout, " ") );
        std::cout << std::endl;
        std::cout << "variance estimate: " << variance_estimate << std::endl;
        std::cout << "smoothness parameter lambda: ";
        for( unsigned int i=0; i<orders.size(); ++i ) {
            std::cout << lambda[orders[i]] << " ";
        }
        std::cout << std::endl;
        std::cout << std::endl;
 
        //         std::cout << "weights: " << std::endl;
        //         for( unsigned int i=0; i<indices.size(); ++i ) {
        //           std::cout << weight_vector[i] << " ";
        // 	}
        //         std::cout << std::endl;
        
        //        if ( indices.front() == 0 ) {
        //            bias = weight_vector(0);
        //            indices.erase( indices.begin() );
        //        } else {
        //            bias = 0.0;
        //        }
       //  
        //        weights.resize( indices.size() );
        //        support_vectors.resize( indices.size(), X.size2() );
        //        for( int i=0; i<indices.size(); ++i ) {
        //            weights( i ) = weight_vector( indices[i] );
        //            ublas::matrix_row<MatT const> X_row( X, indices[i]-1 );  // -1 for bias
        //            ublas::matrix_row<matrix_type> support_vectors_row( support_vectors, i );
        //            support_vectors_row.assign( X_row );
        //        }
        	
        	
 
    }
 
*/





#endif


