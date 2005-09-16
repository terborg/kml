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


#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <iostream>
 
#include <kml/detail/prod_element.hpp>

#include <kml/detail/gemv.hpp>

namespace ublas = boost::numeric::ublas;
 
  
int main(int argc, char *argv[])
{

	ublas::matrix< double > normal_matrix;
	ublas::matrix< ublas::vector<double > > test_matrix;
	
	normal_matrix.resize( 2, 2 );
	test_matrix.resize( 2, 2 );
	
	
	std::cout << test_matrix << std::endl;

	ublas::vector<double> u(2);
	ublas::vector<double> v(2);
	ublas::vector<double> w(2);
	
	u[0] = 1.0;
	u[1] = -1.0;
	v[0] = 0.0;
	v[1] = 2.0;
	w[0] = 0.5;
	w[1] = 0.5;
	
	test_matrix(0,0) = u;
	test_matrix(0,1) = v;
	test_matrix(1,0) = w;
	test_matrix(1,1) = u;
	
	std::cout << "scale_or_dot of " << u << " and " << v << " is " << kml::detail::scale_or_dot( u, v ) << std::endl;
	std::cout << "scale_or_dot of " << 0.2 << " and " << 0.3 << " is " << kml::detail::scale_or_dot( 0.2, 0.3 ) << std::endl;
	std::cout << "scale_or_dot of " << u << " and " << 0.2 << " is " << kml::detail::scale_or_dot( u, 0.2 ) << std::endl;
	std::cout << "scale_or_dot of " << 0.2 << " and " << u << " is " << kml::detail::scale_or_dot( 0.2, u ) << std::endl;

	std::cout << test_matrix << std::endl;

	ublas::vector<double> z = kml::detail::prod_element( u, v );
	std::cout << z << std::endl;
	
	ublas::vector< ublas::vector<double> > test_vector(2);
	kml::detail::gemv( test_matrix, u, test_vector );
	std::cout << "A " << test_matrix << " times " << u << " equals " << test_vector << std::endl;

	kml::detail::gemv( test_matrix, test_vector, u );
	std::cout << "A " << test_matrix << " times " << test_vector << " equals " << u << std::endl;
	

	//kml::detail::gemv( normal_matrix, u, z );
	

	return EXIT_SUCCESS;
}





  
