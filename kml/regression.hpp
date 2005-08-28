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

#ifndef REGRESSION_HPP
#define REGRESSION_HPP

#include <boost/mpl/bool.hpp>

namespace mpl = boost::mpl;

namespace kml {

// define a regression problem

template<typename I, typename O>
class regression {
public:
	typedef regression type;
	typedef I input_type;
	typedef O output_type;
};




// define is_regression so we can recognise the regression type of problems
// for (partial) specialisations

template<typename T>
struct is_regression: mpl::bool_<false> {};

template<typename I, typename O>
struct is_regression<regression<I,O> >: mpl::bool_<true> {};



}



#endif

