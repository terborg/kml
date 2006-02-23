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

#ifndef DETERMINATE_HPP
#define DETERMINATE_HPP


#include <boost/call_traits.hpp>
#include <boost/range/begin.hpp>
#include <boost/range/end.hpp>
#include <boost/ref.hpp>
#include <boost/serialization/access.hpp>

#include <kml/kernel_traits.hpp>
#include <numeric>

// With boost/bind.hpp, the placeholders _1 and _2 are undefined. But why?
#include <boost/lambda/bind.hpp>



namespace lambda = boost::lambda;



namespace kml {

namespace detail {


template<typename Weight, class VecT>
struct multiplies: public std::binary_function<Weight,VecT,VecT> {
	VecT operator()( Weight const &w, VecT const &k ) const {
		return w * k;
	}
};

} // namespace detail







// CachingPolicy!!!

// use boost::enable_if for easy regression / classification selection


// it can be that the kernel returns a double, but that the output type of the kernel machine is binary (classication)
// therefore, Input and Output should be defined

// of the derivatives, however, how is this defined?


/*!
\brief A determinate kernel machine.
\param I the input type
\param O the output type
\param K the kernel type

\ingroup kernel_machines
*/

template< typename I,
	  typename O,
          template<typename,int> class K >
class determinate: public std::unary_function< I,
                                               typename kernel_result<K<I,0> >::type > {
public:
    friend class boost::serialization::access;
    typedef K<I,0> kernel_type;
    typedef I vector_type;
    typedef typename kernel_result<kernel_type>::type result_type;

    // FIXME make this something else...
    typedef double scalar_type;

    template<int N>
    struct kernel_derivative {
       typedef K<I,N> type;
    };
       
    // Bug fixed: use call_traits instead of const&, so can be called with both reference and non-reference
    determinate( typename boost::call_traits<kernel_type>::param_type k ): kernel(k) {}

    // important: needs to be bind(kernel,x,_2) -- as in Ter Borg and Rothkrantz, 2005, equation 1
    //            this is only important in case of asymmetric kernel functions, i.e., those at uneven derivative
    //            orders
    // FIXME optimal return type (by value, return value, i.e. see boost call_traits)
    result_type operator()( typename boost::call_traits<I>::param_type x ) {
        return std::inner_product( weight.begin(),
	                           weight.end(),
				   support_vector.begin(),
				   bias,
                                   std::plus<result_type>(), lambda::bind(detail::multiplies<double,result_type>(), lambda::_1,
                                                             lambda::bind(kernel,x,lambda::_2)) );
    }

    template<class Archive>
    void archive(Archive &archive, const unsigned int version) const {
        archive & kernel;
        archive & bias;
	archive & weight;
	archive & support_vector;
    }

    void clear() {
	bias = 0.0;
	weight.clear();
	support_vector.clear();
    }
    
    kernel_type kernel;
    result_type bias;
    std::vector<double> weight;
    std::vector<I> support_vector;
};





template<class KernelMachine, int N>
class derivative: public std::unary_function<
        typename KernelMachine::argument_type, 
        typename kernel_result<typename KernelMachine::template kernel_derivative<N>::type>::type> {
public:
	
	// this should be done better...
	// template kernel_type<KernelMachine>::type derivative<N>::type kernel_deriv_type;
	// or something like this
	typedef typename KernelMachine::template kernel_derivative<N>::type kernel_type;
	typedef typename kernel_result<typename KernelMachine::template kernel_derivative<N>::type>::type result_type;
	typedef typename KernelMachine::argument_type argument_type;

	derivative( typename boost::call_traits<KernelMachine>::param_type m,
	            typename boost::call_traits<result_type>::param_type bias_init ):
		    machine(boost::cref(m)), kernel(m.kernel.get_parameter()), bias(bias_init) {}
	
        // important: needs to be bind(kernel,x,_2) -- as in Ter Borg and Rothkrantz, 2005, equation 1
        //            this is only important in case of asymmetric kernel functions, i.e., those at uneven derivative
        //            orders
	result_type operator()( argument_type const &x ) const {
		return std::inner_product( boost::begin( machine.get().weight ),
					   boost::end( machine.get().weight ),
		                           boost::begin( machine.get().support_vector ),
					   bias,
                                           std::plus<result_type>(), 
					   bind(detail::multiplies<double,result_type>(),lambda::_1, 
					        lambda::bind(kernel,x,lambda::_2)) );
	}

	kernel_type kernel;
	result_type bias;
	boost::reference_wrapper<KernelMachine const> machine;
};


// use type deduction!!

// FIXME
// this is a good example of an object generator

// template<int N, typename KernelMachine>
// void make_derivative( Kernelmachine const &machine ) {
// 	// figure out bias init!!
// 	return derivative<N,KernelMachine>( machine, some_bias );
// }







} //namespace kml


#endif

