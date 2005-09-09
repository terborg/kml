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

#include <boost/math/special_functions/sinc.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>

#include <kml/regression.hpp>
#include <kml/bindings/fann/ann.hpp>
#include <kml/bindings/fann/rprop.hpp>
#include <kml/bindings/fann/backprop.hpp>
#include <kml/scale.hpp>

namespace fann = kml::bindings::fann;
namespace ublas = boost::numeric::ublas;


int main(int argc, char *argv[]) {
    
    boost::mt19937 randomness;
    boost::normal_distribution<double> norm_dist( 0.0, 0.1 );
    boost::variate_generator<boost::mt19937, boost::normal_distribution<double> > noise( randomness, norm_dist );

    int N = 50;
    ublas::vector<double> y(N);
    std::vector< ublas::vector<double> > x( N );

    for( int i=0; i<N; ++i ) {
        x[i].resize( 1 );
        x[i](0) = (double(i)/double(N-1))*20.0-10.0;
        y[i] = boost::math::sinc_pi(x[i](0)) + noise();


    }

    kml::scale_min_max( x );
    kml::scale_min_max( y );

    std::cout << "Original: " << std::endl;
    std::cout << y << std::endl;

    // create an ANN
    typedef kml::regression< ublas::vector<double>, double > problem;
    unsigned int neurons[3] = {1,10,1};
    fann::ann<problem> my_neural_net( neurons, 3 );

    std::cout << "Training neural network..." << std::endl;
    fann::rprop( x, y, my_neural_net );

    ublas::vector<double> y_test(N);
    std::transform( x.begin(), x.end(), y_test.begin(), my_neural_net );

    std::cout << "Predicted: " << std::endl;
    std::cout << y_test << std::endl;




    return EXIT_SUCCESS;
}






