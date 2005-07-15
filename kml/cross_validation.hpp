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

#ifndef CROSS_VALIDATION_HPP
#define CROSS_VALIDATION_HPP


// old code ! ! And this can be done by mpl code.



namespace kml {


template<class Input, class Output, class KernelMachine>
void cross_validate( Input const &input, Output const &output, KernelMachine &machine ) {



	std::vector< int > indices;
	for( int i=0; i<input.size(); ++i ) indices.push_back(i);

	std::vector<std::vector<double> > svs(4);
	std::vector<std::vector<double> > test_errors(4);
        std::vector<std::vector<double> > train_errors(4);

	int K = 3;
	std::cout << "Performing " << K << "-fold cross-validation..." << std::endl;
	
	for( int i=0; i<100; ++i ) {
		std::cout << std::endl;
		std::cout << "--> run number " << i << std::endl;
		std::cout << std::endl;

		std::random_shuffle( indices.begin(), indices.end() );

		Input train_input( (input.size()*(K-1)/K) );
		Output train_output( (input.size()*(K-1)/K) );
// 		std::cout << "train set size is " << train_input.size() << std::endl;

		for( int j=0; j<(input.size()*(K-1)/K); ++j ) {
/*			std::cout << "train j: " << j << " idx " << indices[j] << std::endl;*/
			train_input[j] = input[indices[j]];
			train_output[j] = output[indices[j]];
		}
		
		Input test_input( input.size() - train_input.size() );
		Output test_output( input.size() - train_input.size() );

		for( int j=(input.size()*(K-1)/K); j<input.size(); ++j ) {
/*			std::cout << "test  j: " << j << " idx " << indices[j] << std::endl;*/
			test_input[j-(input.size()*(K-1)/K)] = input[indices[j]];
			test_output[j-(input.size()*(K-1)/K)] = output[indices[j]];
		}

/*		std::cout << "train set size is " << train_input.size() << std::endl;
		std::cout << "test set size is  " << test_input.size() << std::endl;*/
		ublas::vector<double> test_predict( test_output.size() );
		ublas::vector<double> train_predict( train_output.size() );

// 		print( train_input );
// 		print( train_output );

		std::cout << "training RVM" << std::endl;
		rvm( train_input, train_output, machine );
		std::transform( train_input.begin(), train_input.end(), train_predict.begin(), machine );
		std::transform( test_input.begin(), test_input.end(), test_predict.begin(), machine );
		std::cout << "train error: " << kml::root_mean_square( train_predict - train_output ) << std::endl;
		std::cout << "test error:  " << kml::root_mean_square( test_predict - test_output ) << std::endl;
		std::cout << std::endl;
		train_errors[0].push_back( kml::root_mean_square( train_predict - train_output ) );
		test_errors[0].push_back( kml::root_mean_square( test_predict - test_output ) );
		svs[0].push_back( machine.support_vectors.size() );

		std::cout << "training Figueiredo" << std::endl;
		figueiredo( train_input, train_output, machine );
		std::transform( train_input.begin(), train_input.end(), train_predict.begin(), machine );
		std::transform( test_input.begin(), test_input.end(), test_predict.begin(), machine );
		std::cout << "train error: " << kml::root_mean_square( train_predict - train_output ) << std::endl;
		std::cout << "test error:  " << kml::root_mean_square( test_predict - test_output ) << std::endl;
		std::cout << std::endl;
		train_errors[1].push_back( kml::root_mean_square( train_predict - train_output ) );
		test_errors[1].push_back( kml::root_mean_square( test_predict - test_output ) );
		svs[1].push_back( machine.support_vectors.size() );

		std::cout << "training AO-SVR" << std::endl;
	        typedef kml::aosvr_machine< ublas::vector<double>, kml::gaussian > online_machine_type;
                online_machine_type my_aosvr( machine.kernel, 1.0, 30.0 );
                aosvr_increment( train_input, train_output, my_aosvr );
		std::transform( train_input.begin(), train_input.end(), train_predict.begin(), my_aosvr );
		std::transform( test_input.begin(), test_input.end(), test_predict.begin(), my_aosvr );
		std::cout << "train error: " << kml::root_mean_square( train_predict - train_output ) << std::endl;
		std::cout << "test error:  " << kml::root_mean_square( test_predict - test_output ) << std::endl;
		std::cout << std::endl;
		train_errors[2].push_back( kml::root_mean_square( train_predict - train_output ) );
		test_errors[2].push_back( kml::root_mean_square( test_predict - test_output ) );
		svs[2].push_back( my_aosvr.margin_set.size() );
		
		std::cout << "training SRVM" << std::endl;
		srvm( train_input, train_output, machine );
		std::transform( train_input.begin(), train_input.end(), train_predict.begin(), machine );
		std::transform( test_input.begin(), test_input.end(), test_predict.begin(), machine );
		std::cout << "train error: " << kml::root_mean_square( train_predict - train_output ) << std::endl;
		std::cout << "test error:  " << kml::root_mean_square( test_predict - test_output ) << std::endl;
		std::cout << std::endl;
		train_errors[3].push_back( kml::root_mean_square( train_predict - train_output ) );
		test_errors[3].push_back( kml::root_mean_square( test_predict - test_output ) );
		svs[3].push_back( machine.support_vectors.size() );

	}


        for ( unsigned i=0; i<4; ++i ) {
    	std::cout << "method " << i << std::endl;
    	std::cout << "train error: " << mean( train_errors[i] ) << " +/- " << standard_deviation( train_errors[i] ) << std::endl;
    	std::cout << "test error:  " << mean( test_errors[i] ) << " +/- " << standard_deviation( test_errors[i] ) << std::endl;
	std::cout << "support vec: " << mean( svs[i] ) << " +/- " << standard_deviation( svs[i] ) << std::endl;
    	std::cout << std::endl;
        }
	

}








} // namespace kml






#endif

