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

#ifndef CLASSIFICATION_HPP
#define CLASSIFICATION_HPP


#include <boost/mpl/bool.hpp>

namespace kml {

/*! \brief Defines a classification problem.
	\param I the input type
	\param O the output type

	\sa regression, ranking
	\ingroup problem
*/
template<typename I, typename O>
class classification {
public:
  typedef classification<I,O> type;
  typedef I input_type;
  typedef O output_type;
};



// Define templates is_classification so we can recognise the classification type of problems

template<typename T>
struct is_classification: boost::mpl::bool_<false> {};

template<typename I, typename O>
struct is_classification<classification<I,O> >: boost::mpl::bool_<true> {};


}


#endif

