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

#ifndef RVM_HPP
#define RVM_HPP

#include <boost/call_traits.hpp>
#include <boost/ref.hpp>
#include <boost/range/begin.hpp>
#include <boost/range/end.hpp>
#include <boost/range/size.hpp>
#include <kml/kernel_machine.hpp>
#include <kml/design_matrix.hpp>

// TODO fix old algorithms from kernels.h
#include "kernels.h"



// include AFTER enough traits have been included, of course... (?)
#include <boost/numeric/bindings/atlas/cblas.hpp>
#include <boost/numeric/bindings/atlas/clapack.hpp>




namespace kml {


/*!
\brief Relevance Vector Machine
\param I the input type
\param O the output type
\param K the kernel type
 
Implementation of the Fast RVM [1].
 
\todo
- Cleanup code
- Classification algorithm
- Fix the HtH algorithms
 

\section bibliography References
-# Michael Tipping and Anita Faul. Fast marginal likelihood maximisation for sparse bayesian models. In Cristopher Bishop and Brendan Frey, editors, Proceedings of the Ninth International Workshop on Artificial Intelligence and Statistics, January 3-6 2003, Key West, Florida, 2003. ISBN 0-9727358-0-1.

*/


template< typename Problem, typename Kernel, typename PropertyMap, class Enable = void>
class rvm: public kernel_machine< Problem, Kernel, PropertyMap > {};




template< typename Problem, typename Kernel, typename PropertyMap >
class rvm< Problem, Kernel, PropertyMap, typename boost::enable_if< is_regression<Problem> >::type>:
         public kernel_machine< Problem, Kernel, PropertyMap > {
public:
    typedef kernel_machine< Problem, Kernel, PropertyMap > base_type;
    typedef typename base_type::kernel_type kernel_type;
    typedef typename base_type::result_type result_type;
    typedef typename Problem::input_type input_type;
    typedef typename Problem::output_type output_type;
    typedef double scalar_type;
    typedef ublas::symmetric_matrix<double> symmetric_type;
    typedef ublas::matrix<double> matrix_type;
    typedef ublas::vector<double> vector_type;

    
    /*!	\param k a construction parameter for the kernel type */
    rvm( typename boost::call_traits<kernel_type>::param_type k,
         typename boost::call_traits<PropertyMap>::param_type map ): base_type(k,map) {}

    template< typename TokenIterator >
    rvm( TokenIterator const begin, TokenIterator const end, 
         typename boost::call_traits<kernel_type>::param_type k,
         typename boost::call_traits<PropertyMap>::param_type map ):
	base_type(k,map) {
		// no parameters at the moment
	}


   /*! learn the entire range of keys indicated by this range */
    template<typename KeyIterator>
    void learn( KeyIterator begin, KeyIterator end ) {

	    bool debug = true;
	    	    
	    if (debug)
	    	std::cout << "Computing H..." << std::flush;
	    std::size_t problem_size( end-begin );
	    matrix_type H( problem_size, problem_size + 1 );
	    base_type::design_matrix( begin, end, H );
	    if (debug)
	    	std::cout << "done." << std::endl;

        if (debug)
            std::cout << "computing Hty..." << std::flush;
        vector_type output( problem_size );
        KeyIterator key_iter( begin );
        for( std::size_t i = 0; i<problem_size; ++i ) {
	        output[i] = (*base_type::data)[*key_iter++].get<1>();
        }
        vector_type Hty( problem_size + 1 );
        atlas::gemv( static_cast<matrix_type>(ublas::trans(H)), output, Hty );
        if (debug)
            std::cout << "done." << std::endl;
            
        std::cout << "computing HtH..." << std::flush;
        matrix_type HtH( problem_size + 1, problem_size + 1 );
        atlas::gemm( static_cast<matrix_type>(ublas::trans(H)), H, HtH );
        std::cout << "done." << std::endl;
        
        vector_type diag_HtH( HtH.size1() );
        for( unsigned int i=0; i<HtH.size1(); ++i ) diag_HtH[i] = HtH(i,i);

        
                

        
        // explanation of variables, following Tipping (1999)
        // theta = 1 / alpha

        // create an empty index vector
        std::vector<unsigned int> active_set;
        std::vector<unsigned int> inactive_set( problem_size + 1 );
	for( std::size_t i=0; i<inactive_set.size(); ++i ) inactive_set[i]=i;
        vector_type weight_vector( problem_size + 1 );
        vector_type theta( problem_size + 1 );
        theta.clear();

        // theta is 1/alpha, as in Ralf Herbrich's R implementation
        // it is easier to prune near-zero values than "very large" values

        scalar_type variance_estimate;
        variance_estimate = 0.1;

        // look for the best basis vector we could add
        // If NOT done, e.g., if we would always start with the intercept term
        // or something, this can lead to severe instability in case of an intercept term
        // for which q_i^2 > s_i does not hold. This fixes a bug that occured.
        unsigned int max_idx;
        double max_dl = 0.0;
        for( unsigned int i=0; i<Hty.size(); ++i ) {
            double q_i_2 = std::pow( Hty(i) /variance_estimate,2);
            double s_i   = HtH(i,i)/variance_estimate;
            double delta_l = (q_i_2-s_i)/s_i + std::log( s_i / q_i_2 );
            if (delta_l > max_dl) {
                max_idx = i;
                max_dl = delta_l;
            }
        }
        if (debug)
            std::cout << "starting with point number " << max_idx << "." << std::endl;
        active_set.push_back( max_idx );
        inactive_set.erase( inactive_set.begin() + max_idx );
        theta( max_idx ) = ((Hty(max_idx)*Hty(max_idx)) / HtH(max_idx,max_idx) - variance_estimate) / HtH(max_idx,max_idx);


        
        
        
        vector_type S( problem_size + 1 );
        vector_type Q( problem_size + 1 );
        vector_type mu;

        vector_type residuals( problem_size );

        int best_action = add;  // set to add action (?? ... )

        //
        // MAIN LOOP
        //
        while( best_action != unknown ) {

            /*            std::cout << "active support vectors: ";
                        std::copy( active_set.begin(), active_set.end(), std::ostream_iterator<scalar_type>(std::cout, " ") );
                        std::cout << std::endl; // << "inactive support vectors: ";*/
            //             std::copy( inactive_set.begin(), inactive_set.end(), std::ostream_iterator<scalar_type>(cout, " ") );
            //             std::cout << std::endl;

            // variance changed... recompute sigma and mu


            // TODO don't fully recompute the sigma and mu if the variance did not change enough
            // TODO NEVER recompute the sigma and mu?


            // Prepare the not-yet-inverted part of matrix sigma, and
            // the non-multiplied part of vector mu
            // sigma_inv <- (HtH*(1/var)+A)  (part of eq. 5)
            // mu        <- Hty              (part of eq. 6)


            scalar_type beta = static_cast<scalar_type>(1) / variance_estimate;
            matrix_type sigma_inv( active_set.size(), active_set.size() );
            ublas::symmetric_adaptor< matrix_type > sigma_inv_symm( sigma_inv );

            mu.resize( active_set.size() );
            
            vector_type Hty_cache( active_set.size() );
            matrix_type HtH_cache( HtH.size1(), active_set.size() );
            matrix_type H_cache( H.size1(), active_set.size() );
            
            
            for( unsigned int i=0; i<active_set.size(); ++i ) {
                unsigned int index_1 = active_set[i];
                for( unsigned int j=0; j <= i; ++j ) {
	                unsigned int index_2 = active_set[ j ];
                    sigma_inv_symm(i,j) = beta * HtH(index_1, index_2);
                }
                sigma_inv_symm(i,i) += static_cast<scalar_type>(1) / theta(index_1);
                
                // temporary solution: copy Hty elements to Hty_cache
                // while this should be done with e.g. swap operations
                Hty_cache[i] = Hty[index_1];
                
				// temporary solution: copy column from HtH to HtH_cache
                ublas::column(HtH_cache,i) = ublas::column(HtH,index_1);
                
  				// temporary solution: copy column from HtH to HtH_cache
                ublas::column(H_cache,i) = ublas::column(H,index_1);
                
            }

            // Invert matrix to form matrix sigma and compute vector mu
            // sigma <- (HtH*(1/var)+A)^-1 (equation 5, Tipping, 1999)
            // mu    <- (1/var)*sigma*Hty  (equation 6, Tipping, 1999)
            // TODO symmetric adaptor can go if upper or lower is known (other call to symv)
            // NOTE recip. variance is probably not most efficient like this.

            // create a unit matrix of size active_set.size()
            matrix_type sigma( active_set.size(), active_set.size() );
            sigma.clear();
            for( unsigned int i=0; i<active_set.size(); ++i ) {
                sigma(i,i)=1.0;
            }

            // perform the actual matrix inversion
            ublas::symmetric_adaptor< matrix_type > sigma_symm( sigma );
            atlas::posv( sigma_inv_symm, sigma );

            // update mu vector
            
            // sigma_symm is active_set.size() by active_set.size()
            // Hty_cache is problem_size by active_set.size()
            // mu is active_set size
            
            
            atlas::symv( sigma_symm, Hty_cache, mu );
            atlas::scal( beta, mu );
            
            /*

                COMPUTATIONS DONE TO ESTIMATE THE NEXT BASIS VECTOR

            */
            // work matrix will be HtH.size1() by active_set.size (equals HtH_part size)
            matrix_type work_mat( HtH.size1(), active_set.size() );
            atlas::symm( HtH_cache, sigma_symm, work_mat );

            // compute ALL S(i)'s and Q(i)'s (quite efficient)
            for( unsigned int i=0; i<S.size(); ++i ) {
                S[i] = atlas::dot( ublas::row(work_mat,i), ublas::row(HtH_cache,i) );
            }
            atlas::axpby( beta, diag_HtH, -beta*beta, S );
            atlas::copy( Hty, Q );
            atlas::gemv( -beta*beta, work_mat, Hty_cache, beta, Q );

            // initialise with a non-working move
            best_action = unknown;
            unsigned int action_index = 0;
            scalar_type delta_l_max = 0.0;
            scalar_type theta_l_max = 0.0;

            // loop through the current active set: what are the loglikelihoods of deletion and reestimation?
            for( unsigned int as_i=0; as_i<active_set.size(); ++as_i ) {
                unsigned int i = active_set[as_i];

                // theta is not zero, so S_i != s_i. Equation 23
                scalar_type s_i = S(i) / (1.0 - theta(i) * S(i) );
                scalar_type q_i = Q(i) / (1.0 - theta(i) * S(i) );

                // 		std::cout << "active set entry " << i << std::endl;
                //                 std::cout << "S_i is " << S(i) << ", theta_i is " << theta(i) << ", s_i is " << s_i << std::endl;
                //                 std::cout << "Q_i is " << Q(i) << ", theta_i is " << theta(i) << ", q_i is " << q_i << std::endl;

                // estimate new values for theta, as in equations 20 and 21
                scalar_type q_i_2 = q_i * q_i;
                scalar_type theta_i = ( q_i_2>s_i ? (q_i_2-s_i) / (s_i*s_i) : 0.0 );

                //std::cout << "theta estimate is " << theta_i << std::endl;

                if ( q_i_2 > s_i ) {
                    // reestimate a basis function (equation 32)
                    scalar_type delta_theta = theta_i - theta(i);
                    scalar_type delta_l = (Q(i) * Q(i)) / ( S(i) + 1.0 / delta_theta) - std::log( 1.0 + S(i) * delta_theta );
                    //                 if (debug)
                    //                     std::cout << "reestimating " << i << " had delta l of " << delta_l << std::endl;

                    if ( delta_l > delta_l_max ) {
                        best_action = reestimate;
                        delta_l_max = delta_l;
                        action_index = as_i;
                        theta_l_max = theta_i;
                    }

                    // observe stopping criterion right here (see 4.4 in paper)
                    //                     std::cout << "delta log alpha " << std::log(theta(i)) - std::log( theta_i ) << std::endl;

                } else {
                    // delete a basis function (equation 37)
                    scalar_type delta_l = (Q(i) * Q(i)) / (S(i)-1.0/theta(i)) - std::log( 1.0 - S(i) * theta(i) );
                    //                 if (debug)
                    //                     std::cout << "deleting " << i << " had delta l of " << delta_l << std::endl;

                    if ( delta_l > delta_l_max ) {
                        best_action = remove
                                          ;
                        delta_l_max = delta_l;
                        action_index = as_i;
                        theta_l_max = theta_i;
                    }
                }

            }

            // add?
            for( unsigned int as_i=0; as_i<inactive_set.size(); ++as_i ) {
                unsigned int i = inactive_set[as_i];

                // remark above equation 24: Note that when theta is zero, s_m = S_m and q_m is Q_m.
                scalar_type s_i = S(i);
                scalar_type q_i = Q(i);
                scalar_type q_i_2 = q_i * q_i;
                scalar_type theta_i = (q_i_2-s_i) / (s_i*s_i);

                // could we add?
                if ( q_i_2 > s_i ) {
                    // change in log likelihood, equation 27
                    scalar_type delta_l = (q_i_2-s_i)/s_i + std::log( s_i / q_i_2 );
                    //                 if (debug)
                    //                     std::cout << "adding " << i << " had delta l of " << delta_l << std::endl;

                    if ( delta_l > delta_l_max ) {
                        best_action = add
                                          ;
                        ;
                        delta_l_max = delta_l;
                        action_index = as_i;
                        theta_l_max = theta_i;
                    }
                }
            }

            //if (delta_l_max < std::numeric_limits<double>::epsilon() ) {
            // TODO check this limit!
            if (delta_l_max < 1e-6 ) {
                best_action = unknown;
            }

            if (debug)
                std::cout << "best action was: " << best_action << " with delta_l of " << delta_l_max << std::endl;
            if (debug)
                std::cout << "index was " << action_index << " and theta was " << theta_l_max << std::endl;

                
                
            /*    
                
              BASIS VECTOR HAS BEEN SELECTED; UPDATE REMAINING VALUES
                
                
            */
            // alright, efficiently estimate the new sigma AND mu through update equations
            // on the basis of then newly acquired variance estimate, determine whether we
            // have to recompute sigma and mu

            switch( best_action ) {
            case remove
                        : {
                    // REMOVE, use update equations 38 and 39
                    unsigned int i = active_set[ action_index ];
                    active_set.erase( active_set.begin() + action_index );
                    inactive_set.push_back( i );
                    /*                ublas::matrix_column<ublas::matrix<double, ublas::column_major> > sigma_j( sigma, action_index );*/
                    ublas::matrix_column< matrix_type > sigma_j( sigma, action_index );
                    atlas::axpy( -mu[action_index] / sigma(action_index,action_index), sigma_j, mu );
                    atlas::syr( -1.0/sigma(action_index,action_index), sigma_j, sigma_symm );
                    // NOTE TODO VERY inefficient!
                    preserved_shrink( mu, action_index );
                    preserved_shrink( sigma, action_index, action_index );
                    theta(i) = 0.0;
                    // also remove from H cache
                    remove_column( H_cache, action_index );
                    // also remove from HtH cache...
                    remove_column( HtH_cache, action_index );
                    // also remove from the Hty cache
                    preserved_shrink( Hty_cache, action_index );
                    if (debug)
                        std::cout << "Deleted " << i << " from the active set " << std::endl;
                    break;
                }
            case reestimate: {
                    // RE-ESTIMATE, use update equations 33 and 34
                    unsigned int i = active_set[ action_index ];
                    scalar_type kappa = 1.0 / (sigma(action_index,action_index) + (1.0 / (1.0/theta_l_max - 1.0/theta(i))));
                    /*                ublas::matrix_column<ublas::matrix<double, ublas::column_major> > sigma_j( sigma, action_index );*/
                    ublas::matrix_column< matrix_type > sigma_j( sigma, action_index );
                    atlas::axpy( -kappa * mu[action_index], sigma_j, mu );
                    // FIXED! BUG!! kappa -> -kappa.
                    atlas::syr( -kappa, sigma_j, sigma_symm );
                    theta(i) = theta_l_max;
                    if (debug)
                        std::cout << "Reestimated " << i << std::endl;
                    break;
                }
            case add
                        : {
                    // CAUSE OF BUG: used row(work_mat, action_index) instead of row(work_mat,i)
                    // CAUSE OF BUG: used Q(action_index) instead of Q(i)

                    // ADD, use update equations 28 and 29
                    unsigned int i = inactive_set[ action_index ];
                    inactive_set.erase( inactive_set.begin() + action_index );
                    active_set.push_back( i );

                    // determine the sizes
                    unsigned int old_size = sigma.size1();
                    unsigned int new_size = old_size + 1;

                    // NOTE unclear whether theta_l_max may be used in this case; since at A.2. the
                    // alpha_i is _not_ denoted with a tilde. However, otherwise sigma_ii would always
                    // have value 0.
                    scalar_type sigma_ii = 1.0 / (1.0 / theta_l_max + S(i));
                    scalar_type mu_i = sigma_ii * Q(i);

                    // 		    std::cout << "sigma_ii " << sigma_ii << std::endl;

                    /*		    std::cout << "sigma before: " << sigma << std::endl;*/

                    // perform update of Sigma (equation 28)
                    // TODO FIXME remove beta from the updated matrix? In that case, we don't need to ever
                    // recompute the kernel matrix.
                    atlas::syr( sigma_ii * beta * beta, row(work_mat,i), sigma_symm );
                    preserved_resize( sigma, new_size, new_size );
                    ublas::matrix_vector_slice< matrix_type > sigma_row_part( sigma, ublas::slice(old_size,0,old_size), ublas::slice(0,1,old_size));
                    noalias(sigma_row_part) = -beta * beta * sigma_ii * row(work_mat,i);
                    sigma(old_size,old_size) = sigma_ii;

                    // 		    std::cout << "sigma after: " << sigma << std::endl;


                    // perform update of mu (equation 29)
                    atlas::axpy( -mu_i*beta, row(work_mat,i), mu );
                    preserved_resize( mu, new_size );
                    mu[old_size] = mu_i;

                    theta(i) = theta_l_max;


                    // increase the design matrix
//                     H_cache.resize( H_cache.size1(), H_cache.size2() + 1 );
//                     if( i==0 )
//                         for( unsigned int r=0; r<target.size(); ++r )
//                             H_cache(r,old_size)=1.0;
//                     else
//                         for( unsigned int r=0; r<target.size(); ++r )
//                             H_cache(r,old_size)=base_type::kernel(source[i-1],target[r]);

                    // also increase HtH matrix. Do a preserved resize first
                    // i is the source being added
//                     HtH_cache.resize( HtH_cache.size1(), HtH_cache.size2()+1 );
//                     HtH_comp.fill_column( i, ublas::column( HtH_cache, old_size ) );

                    // increase the Hty cache
//                     Hty_cache.resize( new_size );
//                     Hty_cache[old_size] = Hty[i];

                    if (debug)
                        std::cout << "Added " << i << " to the active set with theta " << theta(i) << std::endl;
                    break;
                }
            default: {
                    break;
                }

            }


            //
            // RE-ESTIMATE VARIANCE
            //


            // recompute Tr(Sigma*var*HtH) = Tr(I) - Tr(Sigma*A)
            scalar_type trace_Sigma_HtH = static_cast<scalar_type>(active_set.size());
            for ( unsigned int i=0; i<active_set.size(); ++i ) {
                trace_Sigma_HtH -= sigma(i,i) / theta(active_set[i]);
                if (debug)
                    std::cout << "sigma_ii " << sigma(i,i) << " theta(i) " << theta(active_set[i]) << "  mu(i) " << mu(i) << std::endl;
            }

            if (debug)
                std::cout << "div: " << (static_cast<scalar_type>(output.size()) - trace_Sigma_HtH ) << std::endl;

            //scalar_type heuh2 = residual_sum_of_squares_2( H, mu, active_set, output ) / (static_cast<scalar_type>(output.size()) - trace_Sigma_HtH );


            atlas::gemv( H_cache, mu, residuals );
            atlas::axpy( -1.0, output, residuals );
            if (debug)
                std::cout << "rss: " << atlas::dot( residuals, residuals ) << std::endl;

            scalar_type heuh2 = atlas::dot( residuals, residuals) / (static_cast<scalar_type>(output.size()) - trace_Sigma_HtH );


            //             std::cout << "variance could be: " << heuh2 << std::endl;
            //             std::cout << "abs-diff-log var: ";
            //             std::cout << std::abs( log(variance_estimate) - log(heuh2) ) << std::endl;

            // NOTE: TODO right here, make decisions about whether or not to recompute the sigma and mu vectors


            variance_estimate = heuh2;

            if (debug)
                std::cout << "variance is now:   " << variance_estimate << std::endl;

            // TODO fit a function to the variance estimate after a number of points?
            // TODO can this be made on-line?



            //             std::cout << "newly estimated mu: " << mu << std::endl;



            // on the basis of abs(diff(log(var))), consider recomputing sigma and mu (and work_mat), or not.



            if (debug) {
                //             int qq;
                //             std::cin >> qq;


            }





        }


        if (debug) {
        	std::cout << "Relevance Vector Machine (Tipping, 2003)" << std::endl;
        	std::cout << "Support vectors:   " << active_set.size() << std::endl;
        	std::cout << "Variance estimate: " << variance_estimate << std::endl;
	}
        /*    std::copy( active_set.begin(),  active_set.end(), std::ostream_iterator<scalar_type>(std::cout, " ") );
            std::cout << std::endl;*/


        std::vector<unsigned int>::iterator bias_loc = std::find( active_set.begin(), active_set.end(), static_cast<unsigned int>(0) );

        // copy bias to machine
        if ( bias_loc != active_set.end() ) {
            base_type::bias = mu( bias_loc - active_set.begin() );
            preserved_shrink( mu, bias_loc - active_set.begin() );
            active_set.erase( bias_loc );
        } else {
            base_type::bias = 0.0;
        }

        // copy support vectors to base type
        //base_type::support_vector.resize( active_set.size() );
        //base_type::weight.resize( active_set.size() );
        //for( unsigned int i=0; i<active_set.size(); ++i ) {
            //         ublas::matrix_row<MatT const> X_row( X, active_set[i]-1 );  // -1 for bias
            //         ublas::matrix_row<matrix_type> support_vectors_row( support_vectors, i );
            //         support_vectors_row.assign( X_row );
            //         weights( i ) = mu( i );
            //base_type::weight[i] = mu( i );
            //base_type::support_vector[i] = source[ active_set[i]-1 ];
        //}










    }






    enum { unknown=0,
           add,
           reestimate,
           remove };




           
           
           
};
















} // namespace kml


#endif

