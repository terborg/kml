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

#ifndef INIT_ZERO_HPP
#define INIT_ZERO_HPP

#include <boost/range/begin.hpp>
#include <boost/range/end.hpp>
#include <boost/range/size.hpp>
#include <boost/range/value_type.hpp>
#include <boost/mpl/if.hpp>
#include <boost/type_traits/is_scalar.hpp>

#include <boost/call_traits.hpp>

namespace mpl = boost::mpl;

namespace kml { namespace detail {

struct zero_element_scalar {
	template<typename T>
	static T compute( T const & ) {
		return static_cast<T>(0);
	}
};

struct zero_element_range {
	template<typename T>
	static T compute( T const &x ) {
		typedef typename boost::range_value<T>::type scalar_type;
		T answer( boost::size(x) );
		std::fill( boost::begin(answer), boost::end(answer), static_cast<scalar_type>(0) );
		return answer;
	}
};

template<typename T>
T zero_element( T const &x ) {
	return mpl::if_< boost::is_scalar<T>, zero_element_scalar, zero_element_range >::type::compute( x );
}

}} // namespace kml::detail

#endif



