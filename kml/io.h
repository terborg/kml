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

#ifndef IO_H
#define IO_H

#include <boost/lambda/lambda.hpp>
#include <boost/range/begin.hpp>
#include <boost/range/end.hpp>
#include <boost/range/value_type.hpp>
#include <iostream>
#include <fstream>

namespace ublas = ::boost::numeric::ublas;
using namespace ::boost::lambda;



template<typename T>
void print_s( T const &v ) {
	std::cout << v << std::endl;
}


template<>
void print_s<ublas::vector<double> >( ublas::vector<double> const &d ) {
	std::for_each( d.begin(), d.end(), std::cout << _1 << " " );
	std::cout << std::endl;
}



template<typename T>
void print( T &t ) {
    typedef typename boost::range_value<T>::type value_type;
    std::for_each( boost::begin(t), boost::end(t), bind( &print_s<value_type>, _1 ) );
    std::cout << std::endl;
}




template<typename T>
void read_data_s( std::ifstream &is, T &d ) {
	T a;
	is >> a;
	d=a;
}


template<>
void read_data_s< ublas::vector<double> >( std::ifstream &is, ublas::vector<double> &d ) {
	for( int i=0; i<d.size(); ++i ) {
		double a;
		is >> a;
		d[i]=a;
	}
}


template<typename T>
void read_data( char* filename, T &t ) {

	std::ifstream file( filename );
	for( int i=0; i< t.size(); ++i ) {
		read_data_s( file, t[i] );
	}
	file.close();

}













#endif


