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

#include <boost/lambda/bind.hpp>
#include <boost/utility/enable_if.hpp>
#include <kml/regression.hpp>
#include <kml/classification.hpp>
#include <kml/ranking.hpp>
#include <vector>
#include <kml/determinate.hpp>


namespace lambda = boost::lambda;


namespace kml {

template< typename Problem, template<typename,int> class Kernel, class Enable = void >
class kernel_machine {};



/*! \brief Regression kernel machine

	This is used as a base for various regression kernel machines, for example kml::rvm.

	\param Problem a regression problem type, for example kml::regression
	\param Kernel kernel to be used by the machine

	\ingroup kernel_machines
*/
template< typename Problem, template<typename,int> class Kernel >
class kernel_machine< Problem, Kernel, typename boost::enable_if< is_regression<Problem> >::type >: 
public std::unary_function< typename Problem::input_type, typename Problem::output_type > {

public:
    typedef Kernel<typename Problem::input_type,0> kernel_type;
    typedef typename Problem::input_type input_type;
    typedef typename Problem::output_type output_type;

    // FIXME make this something else...
    typedef double scalar_type;
       
    // Bug fixed: use call_traits instead of const&, so can be called with both reference and non-reference
	/*! \brief Initializes the kernel
		\param k parameter used to initialize the kernel
	*/
    kernel_machine( typename boost::call_traits<kernel_type>::param_type k ): kernel(k) {}

    // important: needs to be bind(kernel,x,_2) -- as in Ter Borg and Rothkrantz, 2005, equation 1
    //            this is only important in case of asymmetric kernel functions, i.e., those at uneven derivative
    //            orders
    // FIXME optimal return type (by value, return value, i.e. see boost call_traits)
	/*! \brief Evaluates the kernel machine at an input
		\param x the point at which the machine is evaluated
		\return \f$ bias + \sum_i weight[i] \times kernel(x, support\_vector[i]) \f$
	*/
    output_type operator()( typename boost::call_traits<input_type>::param_type x ) {
        return std::inner_product( weight.begin(),
	                           weight.end(),
				   support_vector.begin(),
				   bias,
                                   std::plus<output_type>(), lambda::bind(detail::multiplies<double,output_type>(), lambda::_1,
                                                             lambda::bind(kernel,x,lambda::_2)) );
    }

	/// Clears the machine (operator() it will always returns 0).
    void clear() {
	bias = 0.0;
	weight.clear();
	support_vector.clear();
    }
    
    /// kernel used by the machine
	kernel_type kernel;
	/// bias of the machine
    double bias;
	/// weights of the support vectors
    std::vector<double> weight;
	/// support vectors
    std::vector<input_type> support_vector;
};




/*! \brief Classification kernel machine

	This is used as a base for various classification kernel machines.

	\param Problem a regression problem type, for example kml::classification
	\param Kernel kernel to be used by the machine

	\ingroup kernel_machines
*/
template< typename Problem, template<typename,int> class Kernel>
class kernel_machine<Problem, Kernel, typename boost::enable_if< is_classification<Problem> >::type >: 
public std::unary_function< typename Problem::input_type, typename Problem::output_type > {

public:
    typedef Kernel<typename Problem::input_type,0> kernel_type;
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
                                   std::plus<output_type>(), boost::lambda::bind(detail::multiplies<double,output_type>(), boost::lambda::_1,
                                                             boost::lambda::bind(kernel,x,boost::lambda::_2)) ) >= 0.0;
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


