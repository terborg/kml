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


#ifndef KERNEL_MACHINE_HPP
#define KERNEL_MACHINE_HPP

#include <boost/utility/enable_if.hpp>
#include <kml/regression.hpp>
#include <kml/classification.hpp>
#include <kml/ranking.hpp>


namespace kml {


template< typename Problem, template<typename> class Kernel, class Enable = void >
class kernel_machine {};



// Regression kernel machine

template< typename Problem, template<typename> class Kernel >
class kernel_machine< Problem, Kernel, typename boost::enable_if< is_regression<Problem> >::type >: 
public std::unary_function< typename Problem::input_type, typename Problem::output_type > {

public:
    typedef Kernel<typename Problem::input_type> kernel_type;
    typedef typename Problem::input_type input_type;
    typedef typename Problem::output_type output_type;

    // FIXME make this something else...
    typedef double scalar_type;
       
    // Bug fixed: use call_traits instead of const&, so can be called with both reference and non-reference
    kernel_machine( typename boost::call_traits<kernel_type>::param_type k ): kernel(k) {}

    // important: needs to be bind(kernel,x,_2) -- as in Ter Borg and Rothkrantz, 2005, equation 1
    //            this is only important in case of asymmetric kernel functions, i.e., those at uneven derivative
    //            orders
    // FIXME optimal return type (by value, return value, i.e. see boost call_traits)
    output_type operator()( typename boost::call_traits<input_type>::param_type x ) {
        return std::inner_product( weight.begin(),
	                           weight.end(),
				   support_vector.begin(),
				   bias,
                                   std::plus<output_type>(), bind(detail::multiplies<double,output_type>(), _1,
                                                             bind(kernel,x,_2)) );
    }

    void clear() {
	bias = 0.0;
	weight.clear();
	support_vector.clear();
    }
    
    kernel_type kernel;
    double bias;
    std::vector<double> weight;
    std::vector<input_type> support_vector;
};




// Classification kernel machine

template< typename Problem, template<typename> class Kernel>
class kernel_machine<Problem, Kernel, typename boost::enable_if< is_classification<Problem> >::type >: 
public std::unary_function< typename Problem::input_type, typename Problem::output_type > {

public:
    typedef Kernel<typename Problem::input_type> kernel_type;
    typedef typename Problem::input_type input_type;
    typedef typename Problem::output_type output_type;

    // FIXME make this something else...
    typedef double scalar_type;
       
    // Bug fixed: use call_traits instead of const&, so can be called with both reference and non-reference
    kernel_machine( typename boost::call_traits<kernel_type>::param_type k ): kernel(k) {}

    // FIXME optimal return type (by value, return value, i.e. see boost call_traits)
    output_type operator()( typename boost::call_traits<input_type>::param_type x ) {

	// this must be convertable to bool
        return std::inner_product( weight.begin(),
	                           weight.end(),
				   support_vector.begin(),
				   bias,
                                   std::plus<output_type>(), bind(detail::multiplies<double,output_type>(), _1,
                                                             bind(kernel,x,_2)) ) >= 0.0;
    }

    void clear() {
	bias = 0.0;
	weight.clear();
	support_vector.clear();
    }
    
    kernel_type kernel;
    double bias;

    // the weight should have the sign of the corresponding output sample!
    // w_i = a_i * y_i
    std::vector<double> weight;
    std::vector<input_type> support_vector;
};







}




#endif


