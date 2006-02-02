/***************************************************************************
 *  The Kernel-Machine Library                                             *
 *  Copyright (C) 2004, 2005, 2006 by Rutger W. ter Borg                   *
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


#include <boost/math/special_functions/sinc.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>
#include <kml/gaussian.hpp>
#include <kml/online_svm.hpp>
#include <kml/statistics.hpp>
#include <boost/numeric/ublas/io.hpp>

#include <boost/vector_property_map.hpp>


int main(int argc, char *argv[]) {
    
    boost::mt19937 randomness;
    boost::normal_distribution<double> norm_dist( 0.0, 0.1 );
    boost::variate_generator<boost::mt19937, boost::normal_distribution<double> > noise( randomness, norm_dist );

    typedef std::pair<ublas::vector<double>, double> example_type;
    typedef boost::vector_property_map< example_type > data_type;
    data_type data;

    int N = 50;
    std::vector<ublas::vector<double> > x_vec(N);
    for( int i=0; i<N; ++i ) {
	example_type::first_type x(1);
	x[0] = (double(i)/double(N-1))*20.0-10.0;
        x_vec[i] = x;
        data[i] = std::make_pair( x, boost::math::sinc_pi(x[0]) + noise() );
    }


    std::cout << "Training AO-SVR..." << std::endl;

    typedef kml::regression< example_type > problem_type;
    typedef kml::gaussian< example_type::first_type > kernel_type;
    kml::online_svm< data_type, problem_type, kernel_type > my_machine( 0.1, 10.0, kernel_type(1.6) );
    my_machine.set_data( data );


    std::vector< int > random_order;
    for( int i=0; i<N; ++i ) random_order.push_back(i);

    std::random_shuffle( random_order.begin(), random_order.end() );

    for( int i=0; i<N; ++i )
       my_machine.push_back( random_order[i] );
    std::cout << "Making predictions..." << std::endl;
    ublas::vector<double> y_test(N);
    

    std::transform( x_vec.begin(), x_vec.end(), y_test.begin(), my_machine );

//      std::cout << "RMSE " << kml::root_mean_square( y - y_test ) << std::endl;
//      std::cout << "Cor  " << kml::correlation( y, y_test ) << std::endl;

     std::cout << "Predicted output values:" << std::endl;
     std::cout << y_test << std::endl;

    return EXIT_SUCCESS;
}


