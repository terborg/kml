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
#include <kml/statistics.hpp>
#include <boost/numeric/ublas/io.hpp>

#include <boost/vector_property_map.hpp>
#include <boost/tuple/tuple.hpp>

#include <kml/online_svm.hpp>


int main(int argc, char *argv[]) {

    // create a sinc-data set: sin(x)/x + gaussian noise with sd=0.1
    boost::mt19937 randomness;
    boost::normal_distribution<double> norm_dist( 0.0, 0.1 );
    boost::variate_generator<boost::mt19937, boost::normal_distribution<double> > noise( randomness, norm_dist );

    // define the example type, which is a tuple of {vector,double}, with input=vector, output=double
    typedef boost::tuple<ublas::vector<double>, double> example_type;

    // data are stored in a map, similar to std::map
    // any map type may be used, examples are considered to be referenced by a key-value syntax; keys
    // may be any type; value in this case is the example
    //
    // using integer key types:
    // in case of regression, e.g., data[0] == reference to example 0 == tuple of {vector, double}    
    //            classification, e.g., data[0] == reference to example 0 == tuple of {vector, bool}
    //            single-class, e.g., data[0] == reference to example 0 == tuple of {vector}
    //
    // using pointers: 
    // data[ my_ptr ] = *my_ptr == example pointed to == etc.

    // here, we define the data type, which is a map with integer indexing containing elements of the example_type.
    // in contrast to a std::map, this has O(1) lookup time
    typedef boost::vector_property_map< example_type > data_type;
    data_type data;

    // fill the data set with 50 input-output examples
    // with the inputs in an equally spaced grid [-10,10]
    int N = 50;
    std::vector<ublas::vector<double> > x_vec(N);
    for( int i=0; i<N; ++i ) {
	boost::tuples::element<0,example_type>::type x(1); // equals to "ublas::vector<double> x(1)";
	x[0] = (double(i)/double(N-1))*20.0-10.0;
        x_vec[i] = x;
        data[i] = boost::make_tuple( x, boost::math::sinc_pi(x[0]) + noise() );
    }

    // train the online_svm algorithm on this data set
    std::cout << "Training kernel machine..." << std::endl;

    // define the problem type: regression
    typedef kml::regression< example_type > problem_type;

    // define the kernel type: a gaussian kernel
    typedef kml::gaussian< problem_type::input_type > kernel_type;

    // create the actual kernel machine, an online SVM in this case
    kml::online_svm< problem_type, kernel_type, data_type > my_machine( 10.0, 0.1, kernel_type(1.6), data );

    // train the kernel machine. A kernel machine can be instructed to learn a series of examples 
    // referenced to by a series of keys. E.g., iterators over a std::vector< key_type > of keys would work, but also 
    // more specialised types, such as using lazy-iterator type(s). 

    // we shuffle the keys so that the data is presented in a random way to the kernel machine. Although the 
    // result is the same, we put it here to test just that.
    std::vector< int > random_order;
    for( int i=0; i<N; ++i ) random_order.push_back(i);
    std::random_shuffle( random_order.begin(), random_order.end() );
    my_machine.learn( random_order.begin(), random_order.end() );
    
    // let's see what this gives us...
    std::cout << "Making predictions..." << std::endl;
    ublas::vector<double> y_test(N);
    
    std::transform( x_vec.begin(), x_vec.end(), y_test.begin(), my_machine );
// temp. not working
//    std::cout << "RMSE " << kml::root_mean_square( y - y_test ) << std::endl;
//      std::cout << "Cor  " << kml::correlation( y, y_test ) << std::endl;

     std::cout << "Predicted output values:" << std::endl;
     std::cout << y_test << std::endl;

    return EXIT_SUCCESS;
}





