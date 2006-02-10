/***************************************************************************
 *  The Kernel-Machine Library                                             *
 *  Copyright (C) 2004--2006 by Rutger W. ter Borg                         *
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

#include <boost/program_options.hpp>
#include <iostream>

#include <kml/io.hpp>
#include <kml/gaussian.hpp>

#include <boost/numeric/ublas/vector.hpp>
#include <boost/vector_property_map.hpp>

// include kernel machine types
#include <kml/online_svm.hpp>
#include <kml/rvm.hpp>





namespace po = boost::program_options;
namespace ublas = boost::numeric::ublas;

int main(int argc, char *argv[]) {

	po::options_description descriptions("Usage: kmltool [OPTION]... [FILE]...\n\nWhere OPTION can be one of the following");
	descriptions.add_options()
		( "help", "produce help message" )
		( "machine", po::value<std::string>(), "Selects the kernel machine." )
		( "kernel", po::value< std::string >(), "select the kernel function." )
		( "input-file", po::value< std::string >(), "input file" )
		( "output-file", po::value< std::string >(), "output file" );

	// all positional options should be input files...
	po::positional_options_description pos_options;
	pos_options.add( "input-file", -1 );

	po::variables_map settings;
	po::store( po::command_line_parser( argc,argv ).options(descriptions).positional(pos_options).run(), settings );
	po::notify( settings );

	if ( settings.empty() || settings.count("help") > 0 ) {
		std::cout << descriptions << std::endl;
		return EXIT_SUCCESS;
	}

	if ( settings.count("input-file") == 0 ) {
		std::cout << "Error: no input file given." << std::endl;
		std::cout << descriptions << std::endl;
		return EXIT_SUCCESS;
	}

	if ( settings.count("machine") == 0 ) {
		std::cout << "Error: no --machine option found." << std::endl;
		std::cout << descriptions << std::endl;
		return EXIT_SUCCESS;
	}

	if ( settings.count("kernel") == 0 ) {
		std::cout << "Error: no --kernel option found." << std::endl;
		std::cout << descriptions << std::endl;
		return EXIT_SUCCESS;
	}



	// a whole slew of possible settings...
	enum machine_type { support_vector, relevance_vector };
        enum kernel_type { gaussian, linear };
	machine_type selected_machine( support_vector );
	kernel_type selected_kernel( gaussian );

	//
	// The exact semantics of the command line is a TODO
	//
	std::string machine_string = settings["machine"].as<std::string>();
	boost::char_separator<char> csv_separator(",");
	boost::tokenizer<boost::char_separator<char> > machine_options( machine_string, csv_separator );
	boost::tokenizer<boost::char_separator<char> >::iterator m_o = machine_options.begin();
	if (m_o != machine_options.end() ) {
		if ( *m_o == std::string("svm") ) {
			selected_machine = support_vector;
			std::cout << "support vector machine selected" << std::endl;
			// parse the remaining options later on!!

		}
		if ( *m_o == std::string("rvm") ) {
			selected_machine = relevance_vector;
			std::cout << "relevance vector machine selected" << std::endl;
			// parse any remaining options later on!!
		}
		++m_o;
	}


	std::string kernel_string = settings["kernel"].as<std::string>();
	boost::tokenizer<boost::char_separator<char> > kernel_options( kernel_string, csv_separator );
	boost::tokenizer<boost::char_separator<char> >::iterator k_o = kernel_options.begin();

	if (k_o != kernel_options.end() ) {
		if ( *k_o == std::string("gaussian") ) {
			selected_kernel = gaussian;
			std::cout << "gaussian kernel selected" << std::endl;
		}
		if ( *k_o == std::string("linear") ) {
			selected_kernel = linear;
			std::cout << "gaussian kernel selected" << std::endl;
		}
		++k_o;
	}


	std::string input_file = settings["input-file"].as<std::string>();


	kml::file my_file( input_file );





	switch( my_file.problem_type() ) {
		case kml::io::classification: {
			std::cout << "entering classification part..." << std::endl;
			typedef std::pair< ublas::vector<double>, bool > example_type;
			typedef boost::vector_property_map< std::pair< ublas::vector<double>, bool > > data_type;
			typedef kml::classification< example_type > problem_type;
			data_type data;
			my_file.read( data );

			std::vector<int> learn_keys;
			for( int i=0; i<10; ++i ) {
				learn_keys.push_back( i );
			}
			std::random_shuffle( learn_keys.begin(), learn_keys.end() );

			switch( selected_machine ) {
				case support_vector: {
					switch( selected_kernel ) {
						case gaussian: {
							typedef kml::gaussian< problem_type::input_type > kernel_type;
							typedef kml::online_svm< problem_type, kernel_type, data_type > machine_type;
							kernel_type my_kernel( k_o, kernel_options.end() );
							machine_type my_machine( m_o, machine_options.end(), my_kernel, data );
							my_machine.learn( learn_keys.begin(), learn_keys.end() );
							break;
						}
					}
					break;
				}
				case relevance_vector: {
					switch( selected_kernel ) {
						case gaussian: {
							typedef kml::gaussian< problem_type::input_type > kernel_type;
							kernel_type my_kernel( k_o, kernel_options.end() );

							//typedef kml::rvm< data_type, problem_type, kernel_type > machine_type;

							break;
						}


					} // switch kernel
					break;
				}
			}

			break;
		}
		case kml::io::regression: {
			std::cout << "entering regression part..." << std::endl;
			typedef std::pair< ublas::vector<double>, double > example_type;
			typedef boost::vector_property_map< example_type > data_type;
			typedef kml::regression< example_type > problem_type;

			// translate the file to the data container
			data_type data;
			my_file.read( data );

			std::vector<int> learn_keys;
			for( int i=0; i<5000; ++i ) {
				learn_keys.push_back( i );
			}
			std::random_shuffle( learn_keys.begin(), learn_keys.end() );

			switch( selected_machine ) {
				case support_vector: {
					switch( selected_kernel ) {
						case gaussian: {
							typedef kml::gaussian< problem_type::input_type > kernel_type;
							typedef kml::online_svm< problem_type, kernel_type, data_type > machine_type;
							kernel_type my_kernel( k_o, kernel_options.end() );
							machine_type my_machine( m_o, machine_options.end(), my_kernel, data );
							my_machine.learn( learn_keys.begin(), learn_keys.end() );
							break;
						}

					}
					break;
				}
				case relevance_vector: {
					switch( selected_kernel ) {
						case gaussian: {
							typedef kml::gaussian< problem_type::input_type > kernel_type;
							typedef kml::rvm< problem_type, kernel_type, data_type > machine_type;
							kernel_type my_kernel( k_o, kernel_options.end() );
							machine_type my_machine( m_o, machine_options.end(), my_kernel, data );
							my_machine.learn( learn_keys.begin(), learn_keys.end() );
							break;
						}


					} // switch kernel
					break;
				}
			}


			std::cout << "leaving regression part..." << std::endl;
			break;
		}
		case kml::io::ranking: {
			std::cout << "entering ranking part..." << std::endl;


			break;
		}
		case kml::io::unknown: {
			std::cout << "Sorry, unknown filetype." << std::endl;
		}



	}

	return EXIT_SUCCESS;
}




