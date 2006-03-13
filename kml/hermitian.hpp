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

#ifndef HERMITIAN_HPP
#define HERMITIAN_HPP

#include <boost/call_traits.hpp>
#include <boost/numeric/bindings/atlas/cblas.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/static_assert.hpp>
#include <boost/type_traits.hpp>
#include <kml/input_value.hpp>
#include <kml/distance.hpp>
#include <kml/detail/math.hpp>
#include <kml/power_value.hpp>
#include <cassert>

namespace atlas = boost::numeric::bindings::atlas;

namespace kml { namespace detail {

/*
\brief Statically computes any Hermite "factor". 
This compile-time function computes the constants present a Hermite polynomial H_n(x).
\param N selects Hermite polynomial H_N
\param I selects the I'th constant, those corresponding to lower powers of x^n first
\return A static integer containing the corresponding Hermite "factor".
*/

template<unsigned int N, unsigned int I>
struct hermite_factor {
    static const int value =  detail::power<-1,N/2-I>::value *
                              (factorial<N>::value / (factorial<N-2*(N/2-I)>::value * factorial<N/2-I>::value )) *
                              power<2,N-2*(N/2-I)>::value;
};

template<unsigned int N, int J=N/2>
struct hermite_eval {
    template<typename T>
    static void precompute( T const factor, T const pow_x, T const dot_xx, T *buffer ) {
        *buffer = factor * hermite_factor<N,N/2-J>::value * pow_x;
	hermite_eval<N,J-1>::precompute( factor, pow_x * dot_xx, dot_xx, buffer+1 );
    }
    template<typename T>
    static T compute( T const *buffer, T const pow_x, T const dot_xx ) {
        return *buffer * pow_x + hermite_eval<N,J-1>::compute( buffer+1, pow_x * dot_xx, dot_xx );
    }
};

template<unsigned int N>
struct hermite_eval<N,0> {
    template<typename T>
    static void precompute( T const factor, T const pow_x, T const dot_xx, T *buffer ) {
        *buffer = factor * hermite_factor<N,N/2>::value * pow_x;
    }
    template<typename T>
    static T compute( T const *buffer, T const pow_x, T const dot_xx ) {
        return *buffer * pow_x;
    }
};

// -1, for the case N==1
template<unsigned int N>
struct hermite_eval<N,-1> {
    template<typename T>
    static void precompute( T const factor, T const pow_x, T const dot_xx, T *buffer ) {}
    template<typename T>
    static T compute( T const *buffer, T const pow_x, T const dot_xx ) {
    	// should return 1.0, otherwise the result of this function is indeterminate: can
	// cause bugs. It is much better to not call this function ever at all (current case).
	return 1.0;
    }
};

template<typename Input, int N>
struct hermitian_even {
    static typename input_value<Input>::type compute( typename input_value<Input>::type const *buffer,
						      typename input_value<Input>::type const exp_factor,
                          			      typename input_value<Input>::type const d_square,
							       Input const &u, Input const &v ) {
	return ( *buffer + hermite_eval<N,N/2-1>::compute( buffer+1,d_square,d_square )) *
	          std::exp( exp_factor * d_square );
    }
};

template<class Input, int N>
struct hermitian_uneven {
    static Input compute( typename input_value<Input>::type const *buffer,
      		          typename input_value<Input>::type const exp_factor,
                          typename input_value<Input>::type const d_square,
			  Input const &u, Input const &v ) {
	if (N==1)
		// at N==1, we should not unroll any loop: otherwise we call a termplate with J=-1
		// and since hermite_eval returns a number (is not a void function), the behaviour
		// of this function is WRONG is this if-statement is not here (bug squashed).
		return *buffer * std::exp( exp_factor * d_square ) * (u-v);
	else
		return ( *buffer + hermite_eval<N,N/2-1>::compute( buffer+1,d_square,d_square )) *
		         std::exp( exp_factor * d_square ) * (u-v);
    }
};


} //namespace detail
 




/*!
\brief Hermitian kernel
\param Input defines the underlying input data type
\param N defines the derivative order N, with default 0

This is a template class that creates a function for any derivative of the hermitian kernel. 

It will result in a functor callable by a vector type on which enough operations are defined. It uses template metaprogramming
to perform loop unrollment for the initialisation of any order derivative kernel. Also, the evaluation is also programmed
by loop unrollment. Luckily, size of the loop which is unrolled is only 0.5*O(N), with N derivative order. 
 
Example code:
\code
kml::hermitian< ublas::vector<double> > kernel( 1.0 );
ublas::vector<double> u(2);
ublas::vector<double> v(2);
std::cout << kernel( u, v ) << std::endl;
std::transform( my_data.begin(), my_data.end(), result.begin(), kernel );
std::transform( my_data.begin(), my_data.end(), result.begin(), kml::hermitian< ublas::vector<double>,2 >(1.0) );
\endcode
 

\todo
- call the input type I 
- clean-up
- finish documentation
- complexity guarantees
- loading and saving

*/


template<typename Input, int N=0>
class hermitian: public std::binary_function<Input,
                                            Input,
                                            typename power_value<Input,N>::type> {
public:
    BOOST_SERIALIZATION_SPLIT_MEMBER()
    friend class boost::serialization::access;
    typedef typename input_value<Input>::type scalar_type;
    typedef typename mpl::int_<N>::type derivative_order;
    
    /*! Construct an uninitialised hermitian kernel */
    hermitian() {}
    
    /*! Construct a hermitian kernel
        \param sigma the width of the hermitian kernel */
    hermitian( typename boost::call_traits<scalar_type>::param_type sigma ) {
	set_width( sigma );
    }
    
    /*! \param u input pattern u
        \param v input pattern v
	\return the result of the evaluation of the hermitian kernel for these points. The length of the result depends on 
	        the derivative order of the kernel, which is 1 for N even, and D for N odd, with D the length of the 
		patterns u and v.
    */
    typename power_value<Input,N>::type operator()( Input const &u, Input const &v ) const {
                                  return mpl::if_<
                                         mpl::is_even<mpl::int_<N> >,
                                         detail::hermitian_even<Input,N>,
                                         detail::hermitian_uneven<Input,N> >::type::compute( precomputed_factors,
                                                                                            exp_factor, distance_square(u,v),
                                                                                            u, v );
    }
    
    void set_width( typename boost::call_traits<scalar_type>::param_type sigma ) {
    	assert( sigma > 0.0 );
	width = sigma;
	exp_factor = -1.0 / (2.0*sigma*sigma);
	init();
    }
    
    scalar_type const get_width() const {
    	return width;
    }
    
    void set_scale_factor( typename boost::call_traits<scalar_type>::param_type gamma ) {
        assert( gamma > 0.0 );
	width = std::sqrt(0.5 / gamma);
	exp_factor = -gamma;
	init();
    }
    
    scalar_type const get_scale_factor() const {
    	return -exp_factor;
    }
    
    /*! Initialise the kernel; precompute all needed Hermite factors */
    void init() {

	// precompute, the if statement will be removed by compile-time optimisations
	if ( mpl::is_even< mpl::int_<N> >::type::value ) {
		// herm_factor was: 1.0 / ( sqrt::sqrt(2.0 * sigma ))
		// total_factor was: pow( -herm_factor, N )
        	scalar_type total_factor = detail::power<-1,N/2>::value * std::pow( exp_factor, N/2 );
		precomputed_factors[0] = total_factor * detail::hermite_factor<N,0>::value;
		detail::hermite_eval<N,N/2-1>::precompute( total_factor, -exp_factor, -exp_factor,
		                                           precomputed_factors+1 );
	} else {
		// total_factor was: pow( -herm_factor, N )
		scalar_type total_factor = -std::pow( 1.0 / ( std::sqrt(2.0) * width ), N+1 );
		precomputed_factors[0] = total_factor * detail::hermite_factor<N,0>::value;
 	        detail::hermite_eval<N,N/2-1>::precompute( total_factor, -exp_factor, -exp_factor,
		                                           precomputed_factors+1 );
	}
    }

    template<class Archive>
    void load( Archive &archive, unsigned int const version ) {
    	archive & width;
	exp_factor = -1.0 / (2.0*width*width);
	init();    
    }
    
    template<class Archive>
    void save( Archive &archive, unsigned int const version ) const {
    	archive & width;
    }
    

private:
    scalar_type width;
    scalar_type exp_factor;
    scalar_type precomputed_factors[N/2+1];
};


// for efficiency reasons, a 0th order derivative specialisation

template<typename Input>
class hermitian<Input,0>:public std::binary_function<Input,
                                                    Input,
                                                    typename power_value<Input,0>::type> {
public:
    BOOST_SERIALIZATION_SPLIT_MEMBER()
    friend class boost::serialization::access;
    typedef typename input_value<Input>::type scalar_type;
    typedef typename mpl::int_<0>::type derivative_order;

    /*! Construct an uninitialised hermitian kernel */
    hermitian() {}

    hermitian( typename boost::call_traits<scalar_type>::param_type sigma ) {
       set_width(sigma);
    }

    /*! \param u input pattern u
        \param v input pattern v
	\return the result of the evaluation of the hermitian kernel for these points
    */
    typename power_value<Input,0>::type operator()( Input const &u, Input const &v ) const {
        return std::exp( exp_factor * distance_square( u, v ) );
    }

    void set_width( typename boost::call_traits<scalar_type>::param_type sigma ) {
    	assert( sigma > 0.0 );
	width = sigma;
	exp_factor = -1.0 / (2.0*sigma*sigma);
    }
    
    scalar_type const get_width() const {
    	return width;
    }
    
    void set_scale_factor( typename boost::call_traits<scalar_type>::param_type gamma ) {
        assert( gamma > 0.0 );
	width = std::sqrt(0.5 / gamma);
	exp_factor = -gamma;
    }
    
    scalar_type const get_scale_factor() const {
    	return -exp_factor;
    }
    
    /*! The dimension of the feature space */
    scalar_type const dimension() const {
	return std::numeric_limits<scalar_type>::infinity();
    }

    template<class Archive>
    void load( Archive &archive, unsigned int const version ) {
    	archive & width;
	exp_factor = -1.0 / (2.0*width*width);
    }
    
    template<class Archive>
    void save( Archive &archive, unsigned int const version ) const {
    	archive & width;
    }
    
private:
    scalar_type width;
    scalar_type exp_factor;
};

} // namespace kml


#endif
