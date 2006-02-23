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

#ifndef ONLINE_GREEDY_DETERMINATE_HPP
#define ONLINE_GREEDY_DETERMINATE_HPP

#include <boost/call_traits.hpp>
#include <design_matrix.hpp>
#include <determinate.hpp>
#include <boost/ref.hpp>

#include <boost/range/begin.hpp>
#include <boost/range/end.hpp>
#include <boost/range/const_iterator.hpp>


namespace kml {

template< typename Input,
	  typename Output,
          template<typename,int> class Kernel>
class online_greedy_determinate: public determinate<Input,Output,Kernel> {
public:
        typedef Kernel<Input,0> kernel_type;
        typedef typename kernel_result<kernel_type>::type result_type;
	typedef determinate<Input,Output,Kernel> base_type;
	typedef Input input_type;
	typedef Output output_type;

	online_greedy_determinate( typename boost::call_traits<kernel_type>::param_type k ):
	      base_type(k) {}
	
	// works for any Single Pass Range
	template<typename InputRange, typename OutputRange>
	void learn( InputRange const &input, OutputRange const &output ) {
		typename boost::range_const_iterator<InputRange>::type input_iter( boost::const_begin(input) );
		typename boost::range_const_iterator<OutputRange>::type output_iter( boost::const_begin(output) );
		for( ; input_iter != boost::const_end(input); ++input_iter, ++output_iter )
			push_back( *input_iter, *output_iter );
	}

	// pure virtual
	virtual void push_back( input_type const &input, output_type const &output ) = 0;	

};


}

#endif

