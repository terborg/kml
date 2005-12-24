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

#ifndef ONLINE_DETERMINATE_HPP
#define ONLINE_DETERMINATE_HPP

#include <boost/call_traits.hpp>
#include <boost/range/begin.hpp>
#include <boost/range/end.hpp>
#include <boost/range/const_iterator.hpp>
#include <kml/determinate.hpp>

namespace kml {


/*! Online determinate kernel machine
	\param I the input type
	\param O the output type
	\param K the kernel type

	\ingroup kernel_machines
*/
template< typename Input,typename Output,template<typename,int> class Kernel >
class online_determinate: public determinate<Input,Output,Kernel> {
public:
    typedef Kernel<Input,0> kernel_type;
    typedef typename kernel_result<kernel_type>::type result_type;
    typedef determinate<Input,Output,Kernel> base_type;
    typedef Input input_type;
    typedef Output output_type;

    // constructor
    online_determinate( typename boost::call_traits<kernel_type>::param_type k ):
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

