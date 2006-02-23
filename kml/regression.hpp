/***************************************************************************
 *  The Kernel-Machine Library                                             *
 *  Copyright (C) 2004--2006 by Rutger W. ter Borg                         *
 *                                                                         *
 *  This library is free software; you can redistribute it and/or          *
 *  modify it under the terms of the GNU Lesser General Public             *
 *  License as published by the Free Software Foundation; either           *
 *  version 2.1 of the License, or (at your option) any later version.     *
 *                                                                         *
 *  This library is distributed in the hope that it will be useful,        *
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of         *
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU      *
 *  Lesser General Public License for more details.                        *
 *                                                                         *
 *  You should have received a copy of the GNU Lesser General Public       *
 *  License along with this library; if not, write to the Free Software    *
 *  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  *
 ***************************************************************************/

#ifndef REGRESSION_HPP
#define REGRESSION_HPP

#include <boost/mpl/bool.hpp>
#include <boost/tuple/tuple.hpp>


namespace kml {

/*! \brief Defines a regression problem.

	\param I the input type
	\param O the output type

	\section example Examples
	\code
    typedef kml::regression< ublas::vector<double>, double > problem;
	\endcode
    The problem defined as above can then be used as a template parameter in
	defining kernel machines:
	\code
    kml::rvm< problem, kml::gaussian > my_machine( 1.6 );
	\endcode

	\sa classification, ranking
	\ingroup problem
*/
// template<typename I, typename O>
// class regression {
// public:
// 	typedef regression type;
// 	typedef I input_type;
// 	typedef O output_type;
// };

template<typename T>
class regression {
public:
	typedef regression<T> type;
	typedef T example_type;
	typedef typename boost::tuples::element<0,T>::type input_type;
	typedef typename boost::tuples::element<1,T>::type output_type;
};




// define is_regression so we can recognise the regression type of problems
// for (partial) specialisations

template<typename T>
struct is_regression: boost::mpl::bool_<false> {};

template<typename T>
struct is_regression<regression<T> >: boost::mpl::bool_<true> {};



}



#endif
