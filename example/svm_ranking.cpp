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

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#define BOOST_UBLAS_NESTED_CLASS_DR45

#include <boost/math/special_functions/sinc.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>
#include <kml/gaussian.hpp>
#include <kml/online_ranking_svm.hpp>
#include <kml/statistics.hpp>
#include <boost/numeric/ublas/io.hpp>

int main(int argc, char *argv[]) {
    
    // gcc recommended flags: -O3 -ffast-math -mfpmath=sse -msse2 -march=CPU

    #ifndef __FAST_MATH__
       std::cout << "Enabling -ffast-math is recommended, e.g. std::exp will be twice as fast" << std::endl;
    #endif



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

    std::cout << "Training AO-SVR..." << std::endl;
    kml::online_ranking_svm< ublas::vector<double>, double, kml::gaussian > my_machine( 1.6, 0.1, 10.0 );
/*    my_machine.learn( x, y );

    std::cout << "Making predictions..." << std::endl;
    ublas::vector<double> y_test(N);
    std::transform( x.begin(), x.end(), y_test.begin(), my_machine );

    std::cout << "RMSE " << kml::root_mean_square( y - y_test ) << std::endl;
    std::cout << "Cor  " << kml::correlation( y, y_test ) << std::endl;

    std::cout << "Predicted output values:" << std::endl;
    std::cout << y_test << std::endl;
*/
    return EXIT_SUCCESS;
}


