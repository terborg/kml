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

 
#include <kml/statistics.hpp>
#include <boost/numeric/ublas/vector.hpp> 
#include <boost/numeric/ublas/io.hpp> 
#include <vector> 
#include <list>


namespace ublas = boost::numeric::ublas;


int main(int argc, char *argv[])
{


	ublas::vector<double> rand_vector(100);
	ublas::vector<double> rand_vector_2(100);

	for( int i=0; i<100; ++i ) {
		rand_vector[i] = double(i);
	}

	// answer should be sum(0:99) == 4950
	std::cout << kml::sum( rand_vector ) << std::endl;
	
	// answer should be mean(0:99) = 49.5
	std::cout << kml::mean( rand_vector ) << std::endl;
	
	// answer should be min(0:99) = 0
	std::cout << kml::minimum( rand_vector ) << std::endl;
	
	// answer sould be max(0:99) = 99
	std::cout << kml::maximum( rand_vector ) << std::endl;
	
	// population variance should be 841.6666667
	std::cout << kml::variance( rand_vector ) << std::endl;
	std::cout << kml::standard_deviation( rand_vector ) << std::endl;
	std::cout << kml::mean_square( rand_vector ) << std::endl;
	std::cout << kml::root_mean_square( rand_vector ) << std::endl;

	// but, sum should also work on a vector!
	std::vector< ublas::vector<double> > vec_of_vec;
	std::list< ublas::vector<double> > list_of_vec;

	for( int i=0; i<100; ++i ) {
		vec_of_vec.push_back( rand_vector );
		list_of_vec.push_back( rand_vector );
	}
	
	// answer should be a vector of lenght 100 containing 0,100,...,9900
	std::cout << kml::sum( vec_of_vec ) << std::endl;

	// answer should be a vector of lenght 100 containing 0,1,...,99
	std::cout << kml::mean( vec_of_vec ) << std::endl;

	// answer should be min(0:99) = (0:99)
	std::cout << kml::minimum( vec_of_vec ) << std::endl;
	std::cout << kml::variance( vec_of_vec ) << std::endl;
	
	vec_of_vec.push_back( rand_vector_2);
	std::cout << kml::minimum( vec_of_vec ) << std::endl;
	std::cout << kml::maximum( vec_of_vec ) << std::endl;
	std::cout << kml::variance( vec_of_vec ) << std::endl;
	std::cout << kml::standard_deviation( vec_of_vec ) << std::endl;
	std::cout << kml::root_mean_square( rand_vector ) << std::endl;

	std::cout << "List container: " << std::endl;	
	std::cout << kml::minimum( list_of_vec ) << std::endl;


	return EXIT_SUCCESS;
}


 
 
 
 
 
 
 
 
 
 


