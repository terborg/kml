/*****************************************************************************
 *  The Kernel-Machine Library                                               *
 *  Copyright (C) 2004, 2005 by Rutger W. ter Borg and Meredith L. Patterson *
 *                                                                           *
 *  This program is free software; you can redistribute it and/or            *
 *  modify it under the terms of the GNU General Public License              *
 *  as published by the Free Software Foundation; either version 2           *
 *  of the License, or (at your option) any later version.                   *
 *                                                                           *
 *  This program is distributed in the hope that it will be useful,          *
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of           *
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the            *
 *  GNU General Public License for more details.                             *
 *                                                                           *
 *  You should have received a copy of the GNU General Public License        *
 *  along with this program; if not, write to the Free Software              *
 *  Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307    *
 *****************************************************************************/

#ifndef RANKING_HPP
#define RANKING_HPP

#include <boost/mpl/bool.hpp>
#include <boost/tuple/tuple.hpp>

namespace kml {

/*! \brief Defines a ranking problem.
	\param I the input type
	\param O the output type

	\sa regression, classification
	\ingroup problem
*/

template<typename T>
class ranking {
public:
	typedef ranking<T> type;
	typedef T example_type;
	typedef typename boost::tuples::element<0,T>::type input_type;
	typedef typename boost::tuples::element<1,T>::type output_type;
};

// Define templates is_ranking so we can recognise the ranking type of problems

template<typename T>
struct is_ranking: boost::mpl::bool_<false> {};

template<typename T>
struct is_ranking<ranking<T> >: boost::mpl::bool_<true> {};

}

#endif

