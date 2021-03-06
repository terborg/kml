/***************************************************************************
 *  The Kernel-Machine Library                                             *
 *  Copyright (C) 2005, 2006 by Meredith L. Patterson                      *
 *  Copyright (C) 2005, 2006 by Rutger W. ter Borg                         *
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

#ifndef RANKING_HPP
#define RANKING_HPP

#include <boost/mpl/bool.hpp>
#include <boost/tuple/tuple.hpp>

namespace kml {

/*! \brief Defines a ranking problem.
	\param T the example type

The example type should be a boost::tuple defining the input as the first
element, the group as the second element, and the output as the third element.

	\sa regression, classification
	\ingroup problem
*/

template<typename T>
class ranking {
public:
	typedef ranking<T> type;
	typedef T example_type;
	typedef typename boost::tuples::element<0,T>::type input_type;
        typedef typename boost::tuples::element<1,T>::type group_type;
	typedef typename boost::tuples::element<2,T>::type output_type;
};

// Define templates is_ranking so we can recognise the ranking type of problems

template<typename T>
struct is_ranking: boost::mpl::bool_<false> {};

template<typename T>
struct is_ranking<ranking<T> >: boost::mpl::bool_<true> {};

}

#endif

