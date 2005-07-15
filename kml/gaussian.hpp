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

#ifndef GAUSSIAN_HPP
#define GAUSSIAN_HPP

#include <boost/call_traits.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/static_assert.hpp>
#include <boost/type_traits.hpp>
#include <kml/linear.hpp>
#include <kml/distance.hpp>
#include <kml/math.hpp>



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

template<class Range, int N>
struct gaussian_even {
    static typename boost::range_value<Range>::type compute( typename scalar_type<Range>::type const *buffer,
							      typename scalar_type<Range>::type const exp_factor,
							      Range const &u, Range const &v ) {
	double d_square( distance_square(u,v) );
	return ( *buffer + hermite_eval<N,N/2-1>::compute( buffer+1,d_square,d_square )) *
	          std::exp( exp_factor * d_square );
    }
};

template<class Range, int N>
struct gaussian_uneven {
    static Range compute( typename scalar_type<Range>::type const *buffer,
      		           typename scalar_type<Range>::type const exp_factor,
			   Range const &u, Range const &v ) {
	double d_square( distance_square(u,v) );
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
\brief Gaussian kernel
\param T defines the underlying input data type
\param N defines the derivative order N, with default 0

This is a template class that creates a function for any derivative of the Gaussian kernel. 

It will result in a functor callable by a vector type on which enough operations are defined. It uses template metaprogramming
to perform loop unrollment for the initialisation of any order derivative kernel. Also, the evaluation is also programmed
by loop unrollment. Luckily, size of the loop which is unrolled is only 0.5*O(N), with N derivative order. 
 
Example code:
\code
kml::gaussian< ublas::vector<double> > kernel( 1.0 );
ublas::vector<double> u(2);
ublas::vector<double> v(2);
std::cout << kernel( u, v ) << std::endl;
std::transform( my_data.begin(), my_data.end(), result.begin(), kernel );
std::transform( my_data.begin(), my_data.end(), result.begin(), kml::gaussian< ublas::vector<double>,2 >(1.0) );
\endcode
 

\todo
- call the input type I 
- clean-up
- finish documentation
- complexity guarantees

*/


template<typename Range, int N=0>
class gaussian: public std::binary_function<Range,
                                            Range,
                                            typename kml::power_return_type<Range,N>::type> {
public:
    BOOST_SERIALIZATION_SPLIT_MEMBER();
    friend class boost::serialization::access;
    typedef typename boost::range_value<Range>::type scalar_type;
    typedef typename mpl::int_<N>::type derivative_order;

    
    /*! Construct an uninitialised Gaussian kernel */
    gaussian() {}
    
    /*! Construct a Gaussian kernel
        \param sigma the width of the Gaussian kernel */
    gaussian( typename boost::call_traits<scalar_type>::param_type sigma ): parameter(sigma) {
	set_parameter( sigma );
    }

    
    /*! \param u input pattern 1
        \param v input pattern 2
	\return the result of the evaluation of the Gaussian kernel for these points. The length of the result depends on 
	        the derivative order of the kernel, which is 1 for N even, and D for N odd, with D the length of the 
		patterns u and v.
    */
    typename kml::power_return_type<Range,N>::type operator()( Range const &u, Range const &v ) const {
                                  return mpl::if_<
                                         mpl::is_even<mpl::int_<N> >,
                                         detail::gaussian_even<Range,N>,
                                         detail::gaussian_uneven<Range,N> >::type::compute( precomputed_factors,
                                                                                            exp_factor, u, v );
    }
    
    void set_parameter( typename boost::call_traits<scalar_type>::param_type sigma ) {
	exp_factor = -1.0 / (2.0*sigma*sigma);

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
		scalar_type total_factor = -std::pow( 1.0 / ( std::sqrt(2.0) * sigma ), N+1 );
		precomputed_factors[0] = total_factor * detail::hermite_factor<N,0>::value;
 	        detail::hermite_eval<N,N/2-1>::precompute( total_factor, -exp_factor, -exp_factor,
		                                           precomputed_factors+1 );
	}
    }

    scalar_type const &get_parameter() const {
    	return parameter;
    }
        
    template<class Archive>
    void load( Archive &archive, unsigned int const version ) {
    	archive & parameter;
	set_parameter(parameter);    
    }
    
    template<class Archive>
    void save( Archive &archive, unsigned int const version ) const {
    	archive & parameter;
    }
    

private:
    scalar_type parameter;
    scalar_type exp_factor;
    scalar_type precomputed_factors[N/2+1];
};


// for efficiency reasons, a 0th order derivative specialisation
// This is also the only Mercer Kernel of the series.

template<typename Range>
class gaussian<Range,0>:public std::binary_function<Range,
                                                    Range,
                                                    typename kml::power_return_type<Range,0>::type> {
public:
    BOOST_SERIALIZATION_SPLIT_MEMBER();
    friend class boost::serialization::access;
    typedef typename boost::range_value<Range>::type scalar_type;
    typedef typename mpl::int_<0>::type derivative_order;

    gaussian() {}
    gaussian( typename boost::call_traits<scalar_type>::param_type sigma ): parameter(sigma) {
       set_parameter(sigma);
    }

    typename kml::power_return_type<Range,0>::type operator()( Range const &u, Range const &v ) const {
        return std::exp( exp_factor * distance_square( u, v ) );
    }
    
    void set_parameter( typename boost::call_traits<scalar_type>::param_type sigma ) {
	exp_factor = -1.0/(2.0*sigma*sigma);
    }

    scalar_type const &get_parameter() const {
    	return parameter;
    }

    /*! The dimension of the feature space */
    scalar_type const dimension() const {
	return std::numeric_limits<scalar_type>::infinity();
    }

    template<class Archive>
    void load( Archive &archive, unsigned int const version ) {
    	archive & parameter;
	set_parameter(parameter);
    }
    
    template<class Archive>
    void save( Archive &archive, unsigned int const version ) const {
    	archive & parameter;
    }
    
private:
    scalar_type parameter;
    scalar_type exp_factor;
};

} // namespace kml


#endif
