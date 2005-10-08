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




#ifndef FIGUEIREDO_HPP
#define FIGUEIREDO_HPP

#include <ext/numeric>
#include "least_squares.h"

#include <design_matrix.hpp>

#include <probabilistic.hpp>
#include <boost/numeric/bindings/traits/ublas_matrix.hpp>



namespace kml {


/*!
\brief Adaptive Sparseness using Jeffreys' prior
\param I the input type
\param O the output type
\param K the kernel type

Two update rules are computed here:
 
- \f$ \sigma^2_{n+1} = \Vert Hx-y\Vert^2 / N \f$
- \f$ x_{n+1} = (\vert Ix^{-2}\vert H)^{-1}H^Ty \f$
     
The model is initialised with a ridge regression estimate.
 
However, the original system of update equations has no computational problems at all,
despite mentioned as such.


Example:
\code
kml::figueiredo< ublas::vector<double>, double, kml::gaussian > my_machine( 1.6 );
my_machine.learn( x, y );
\endcode
  
  

\todo
- Remove dependency of LAPACK, move to ATLAS' posv
- Add a true probabilistic framework

\section bibliography References

-# Adaptive Sparseness using Jeffreys' prior
   Mario A.T. Figueiredo, 2003

*/



template<typename I, typename O, template<typename,int> class K >
class figueiredo: public probabilistic<I,O,K> {
public:
    typedef probabilistic<I,O,K> base_type;
    typedef typename base_type::kernel_type kernel_type;
    typedef typename base_type::result_type result_type;
    typedef I input_type;
    typedef O output_type;
    typedef double scalar_type;
    typedef ublas::symmetric_matrix<double> symmetric_type;
    typedef ublas::matrix<double> matrix_type;
    typedef ublas::vector<double> vector_type;

    /*! \param k the kernel construction parameter */
    figueiredo( typename boost::call_traits<kernel_type>::param_type k ): base_type(k) {}
    

    template<class IRange, class ORange>
    void learn( IRange const &input, ORange const &output ) {
		learn( input, input, output );
    }

    template<class IRange, class ORange>
    void learn( IRange const &source, IRange const &target, ORange const &output ) {


    // create the kernel matrix (symmetric part)
    // initialise & compute H, Hty and HtH
    matrix_type H;
    // TODO traits form base_type::kernel!!
    design_matrix( source, target, base_type::kernel, 1.0, H );

    // TODO use a less heavy call?
    vector_type Hty( source.size()+1 );
    atlas::gemv( CblasTrans, 1.0, H, output, 0.0, Hty );    
    
    
    symmetric_type HtH( H.size2(), H.size2() );
    inner_prod( H, HtH );

    // indices <- 0,1,2,...,N-1
    vector_type weight_vector( HtH.size1() );
    std::vector<unsigned int> indices( HtH.size1() );
    iota( indices.begin(), indices.end(), 0 );

    // Figueiredo's method should start with a least squares estimate or an ridge regression estimate
    // start with a ridge regression estimate (needed in case of exact data), with a value of
    // lambda which prevents singularities.
    // TODO figure out why this formulation solves the condition number thing.
    ridge_regression( HtH, weight_vector, Hty, std::numeric_limits<scalar_type>::epsilon() *
                      static_cast<scalar_type>( Hty.size() * Hty.size() * Hty.size() ) );

    // Figueiredo's method main loop
    // Stopping criterion: minimum change in log weight must be larger than 1e-6.
    // initialise the maximum difference with something that is larger than the stopping criterion
    scalar_type variance_estimate(0);
    scalar_type max_diff = static_cast<scalar_type>(1);
    while( max_diff > static_cast<scalar_type>(1e-6) ) {

        // discard active variables with a weight smaller than the machine epsilon
        prune_indices( indices, weight_vector );

        // Estimate the variance
        // Since the variance is added to the diagonal of the system matrix, also do some
        // singularity prevention, by setting the minimum variance estimate to N^3 * machine epsilon
	
	// residual_sum_squares will become mean_dot( Hw - y )
        variance_estimate = std::max<scalar_type>( residual_sum_squares( H, weight_vector, indices, output )
	                                           / static_cast<scalar_type>(output.size()),
                                      std::numeric_limits<scalar_type>::epsilon() *
                                      static_cast<scalar_type>( indices.size() * indices.size() * indices.size()) );

        // TODO: improve naming scheme: index_sv, i_idx? A, B? indices? etc.
        // Lapack: requires column major matrices
        ublas::symmetric_matrix<double, ublas::lower, ublas::column_major> A( indices.size(), indices.size() );
        ublas::matrix<double, ublas::column_major> B( indices.size(), 1 );


        // NOTE --> GENERIC: solve a dynamic ridge-like system (as in RVM)
        // A <- I*var*diag(weight^-2) + HtH
        // B <- Ht*y
        for( unsigned int i=0; i<indices.size(); ++i ) {
            unsigned int index_sv = indices[i];
            B(i,0) = Hty( index_sv );
            for( unsigned int col=0; col <= i; ++col ) {
                A(i,col) = HtH( index_sv, indices[col] );
            }
            A(i,i) += variance_estimate / ( weight_vector(index_sv) * weight_vector(index_sv) );
        }

        // Solve the linear system of equations
        // B <- (HtH + I*var*B^-2)^-1*Ht*y
        lapack::ppsv( A, B );

        max_diff = static_cast<scalar_type>(0.0);
        for( unsigned int i=0; i<indices.size(); ++i ) {
            unsigned int i_idx = indices[i];
            scalar_type weight_old = weight_vector(i_idx);
            weight_vector(i_idx) = B(i,0);
            scalar_type diff = std::fabs( std::log(fabs(weight_old)) - std::log(fabs(weight_vector(i_idx))) );
            if (diff > max_diff)
                max_diff = diff;
        }
    }
    // that's all, folks!

    std::cout << "Adaptive Sparseness using Jeffreys Prior (Figueiredo, 2003)" << std::endl;
    std::cout << "Support vectors:   " << indices.size() << std::endl;
    std::cout << "variance estimate: " << variance_estimate << std::endl;


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

}; // class figueiredo

} // namespace kml


#endif
