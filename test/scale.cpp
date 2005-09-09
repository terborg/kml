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


#include <vector>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <kml/io.hpp>
#include <kml/statistics.hpp>
#include <kml/scale.hpp>


namespace ublas = boost::numeric::ublas;


int main(int argc, char *argv[])
{

  std::vector<ublas::vector<double> > points;
  std::vector<bool> target;


  //kml::read_svm_light( "data/pos_symmetry.data", points, target);
  kml::read_svm_light( "data/heart_scale.data", points, target);
  std::cout << "Read inputs " << std::endl;
	
   std::cout << "Minima: " << kml::minimum( points ) << std::endl;
   std::cout << "Mean:   " << kml::mean( points ) << std::endl;
   std::cout << "sd:     " << kml::standard_deviation( points ) << std::endl;
   std::cout << "Maxima: " << kml::maximum( points ) << std::endl;
   std::cout << std::endl;

   //std::cout << "Reciprocal of maxima: " << kml::detail::reciprocal_vector( kml::maximum( points ) ) << std::endl;


   kml::scale_min_max( points );

   std::cout << "Min max:" << std::endl;
   std::cout << "Minima: " << kml::minimum( points ) << std::endl;
   std::cout << "Mean:   " << kml::mean( points ) << std::endl;
   std::cout << "sd:     " << kml::standard_deviation( points ) << std::endl;
   std::cout << "Maxima: " << kml::maximum( points ) << std::endl;
   std::cout << std::endl;

   kml::scale_mean_sd( points );

   std::cout << "Scaled: " << std::endl;
   std::cout << "Minima: " << kml::minimum( points ) << std::endl;
   std::cout << "Mean:   " << kml::mean( points ) << std::endl;
   std::cout << "sd:     " << kml::standard_deviation( points ) << std::endl;
   std::cout << "Maxima: " << kml::maximum( points ) << std::endl;
   std::cout << std::endl;

}
















