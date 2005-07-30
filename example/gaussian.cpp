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


#include <kml/gaussian.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <iostream>

namespace ublas = boost::numeric::ublas;

using namespace std;

int main(int argc, char *argv[])
{

  // vector input types
  ublas::vector<double> u(2);
  ublas::vector<double> v(2);
  u[0] = 1.0;
  u[1] = 2.0;
  v[0] = 3.0;
  v[1] = 4.0;
  kml::gaussian< ublas::vector<double> > kernel( 1.0 );
  std::cout << kernel( u, v ) << std::endl;
  
  // scalar input types
  std::cout << kml::gaussian< float >( 1.0 )( 1.5, 2.0 ) << std::endl;

  return EXIT_SUCCESS;
}








