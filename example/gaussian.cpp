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

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <kml/gaussian.hpp>
#include <fstream>
#include <iostream>

namespace ublas = boost::numeric::ublas;

int main(int argc, char *argv[]) {

  // vector input types
  ublas::vector<double> u(2);
  ublas::vector<double> v(2);
  u[0] = 1.0;
  u[1] = 2.0;
  v[0] = 3.0;
  v[1] = 4.0;
  kml::gaussian< ublas::vector<double> > kernel( 1.0 );
  std::cout << kernel( u, v ) << std::endl;
  
  // query its parameter in different formats
  std::cout << "Kernel width (sigma):        " << kernel.get_width() << std::endl;
  std::cout << "Kernel scale factor (gamma): " << kernel.get_scale_factor() << std::endl;
  
  // inline use on scalar input types
  std::cout << kml::gaussian< float >( 1.0 )( 1.5, 2.0 ) << std::endl;
  
  // save the kernel
  std::ofstream ofs("testfile.txt");
  boost::archive::text_oarchive my_oarchive( ofs );
  my_oarchive << kernel;
  ofs.close();
  std::cout << "Saved: " << kernel << std::endl;
  
  // ... sometime later, load the kernel
  kml::gaussian< ublas::vector<double> > new_kernel;
  std::ifstream ifs("testfile.txt");
  boost::archive::text_iarchive my_iarchive( ifs );
  my_iarchive >> new_kernel;
  ifs.close();
  std::cout << "Loaded: " << new_kernel << std::endl;
  
  return EXIT_SUCCESS;
}

