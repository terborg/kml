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

#include <kml/polynomial.hpp>

#include <boost/numeric/bindings/ublas/matrix.hpp>
#include <boost/numeric/bindings/ublas/vector.hpp>

#include <vector>

#include <boost/random/variate_generator.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_real.hpp>


#include <kml/incomplete_cholesky.hpp>
#include <kml/kernel_machine.hpp>
#include <kml/io.hpp>



#include <boost/property_map/vector_property_map.hpp>



namespace ublas = boost::numeric::ublas;

int main(int argc, char *argv[]) {
    boost::mt19937 randomness;

    typedef boost::uniform_real<double> dist_type;
    dist_type random_dist( 0.0, 1.0 );


    boost::variate_generator<boost::mt19937, dist_type > noise( randomness, random_dist );

    typedef boost::tuple<ublas::vector<double>, double> example_type;
    typedef kml::regression< example_type > problem_type;

    typedef boost::vector_property_map< boost::tuple< ublas::vector<double>, bool> > data_access_type;
    data_access_type data;

    typedef kml::polynomial< ublas::vector<double> > kernel_type;
    kernel_type my_kernel( 1.0, 0.0, 3.0 );

    int N = 1000;

    std::vector<int> all_keys;
    for ( int sample = 0; sample < N; ++sample )
        all_keys.push_back( sample );

    // Create random data
    for ( int sample = 0; sample < N; ++sample ) {
        int data_dim = 10;
        ublas::vector<double> x( data_dim );
        for( int i=0; i<data_dim; ++i ) {
            x[i] = noise() / double(data_dim);
        }
        data[sample] = x;
        //std::cout << x << std::endl;
    }

    kml::incomplete_cholesky<kernel_type, data_access_type> my_eval( my_kernel, data );
    kml::incomplete_cholesky<kernel_type, data_access_type> my_eval2( my_kernel, data );

    my_eval.compute_R( all_keys.begin(), all_keys.begin() + 8, 3 );
    my_eval2.compute_R( all_keys.begin(), all_keys.begin() + 5, 3 );

    my_eval2.increment( 5 );
    my_eval2.increment( 6 );
    my_eval2.increment( 7 );

    std::cout << "Pivots used: ";
    for( int i=0; i< my_eval.basis_size; ++i )
        std::cout << my_eval.pivot[i] << " ";
    std::cout << std::endl;
    std::cout << my_eval.RT.view() << std::endl;

    std::cout << "Pivots-2 used: ";
    for( int i=0; i< my_eval2.basis_size; ++i )
        std::cout << my_eval2.pivot[i] << " ";
    std::cout << std::endl;
    std::cout << my_eval2.RT.view() << std::endl;


    return EXIT_SUCCESS;
}






