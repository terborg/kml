/***************************************************************************
 *  FANN bindings                                                          *
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

#ifndef BACKPROP_HPP
#define BACKPROP_HPP

#include <boost/range/begin.hpp>
#include <boost/range/size.hpp>

#include <kml/bindings/fann/ann.hpp>
#include <vector>


namespace kml { namespace bindings { namespace fann {


template<typename I,typename O,typename ANN>
void backprop( I &inputs, O &outputs, ANN &result_ann ) {

	fann_set_training_algorithm( result_ann.m_fann, FANN_TRAIN_BATCH );
	
	fann_train_data m_data;
	m_data.num_data = boost::size( inputs );
	m_data.num_input = boost::size( inputs.front() );
	m_data.num_output = 1;				// scalar output assumed at the moment

	m_data.input = new double*[inputs.size()];
	m_data.output = new double*[outputs.size()];

	for( unsigned int i=0; i<inputs.size(); ++i ) {
		m_data.input[i] = &inputs[i][0];
		m_data.output[i] = &outputs[i];
	}
	
	fann_train_on_data( result_ann.m_fann, &m_data, 100000, 100, 1e-5 );

	delete m_data.input;
	delete m_data.output;
}

}}}

#endif

