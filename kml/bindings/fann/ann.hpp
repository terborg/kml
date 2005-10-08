/***************************************************************************
 *  FANN bindings                                                          *
 *  Copyright (C) 2005 by Rutger W. ter Borg                               *
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

#ifndef ANN_HPP
#define ANN_HPP

#include <boost/range/begin.hpp>
#include <boost/range/size.hpp>
#include <boost/type_traits.hpp>


namespace kml { namespace bindings { namespace fann {

// include within this namespace, otherwise we get name clashes
#include <doublefann.h>



template<typename Problem>
class ann: 
public std::unary_function< typename Problem::input_type, typename Problem::output_type > {
public:
    typedef typename Problem::input_type input_type;
    typedef typename Problem::output_type output_type;

        ann( unsigned int* neurons, unsigned int nr_of_neurons, double connection_rate=1.0, double learning_rate=0.7 ) {
		// should be done by fann create array
		// fann_create_array( 1.0, 0.7, 3
		m_fann = fann_create_array( connection_rate, learning_rate, nr_of_neurons, neurons );

		fann_set_activation_function_hidden( m_fann, FANN_SIGMOID );
		fann_set_activation_function_output( m_fann, FANN_SIGMOID );

		//fann_print_parameters( m_fann );
	}

	~ann() {
		//fann_destroy(m_fann);
	}
	
	double operator()( input_type &input ) {
	
		// the answer will be returned (and allocated elsewhere)
		double *answer = fann_run( m_fann, &input[0] );
		return answer[0];
	}


	// local memory: the fann
	fann *m_fann;
};

}}}

#endif




