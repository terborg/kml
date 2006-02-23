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

#ifndef MAX_ELEMENT_HPP
#define MAX_ELEMENT_HPP

#include <boost/range/begin.hpp>
#include <boost/range/end.hpp>
#include <boost/range/size.hpp>
#include <boost/range/value_type.hpp>
#include <boost/mpl/if.hpp>
#include <boost/type_traits/is_scalar.hpp>
#include <boost/bind.hpp>

namespace mpl = boost::mpl;

namespace kml { namespace detail {

struct max_element_scalar {
	template<typename T>
	static T compute( T const &x, T const &y ) {
		return std::max<T>( x, y );
	}
};

struct max_element_range {
	template<typename T>
	static T compute( T const &x, T const &y ) {
		typedef typename boost::range_value<T>::type scalar_type;
		T answer( boost::size(x) );
		std::transform( boost::begin(x), boost::end(x), boost::begin(y), boost::begin(answer), boost::bind( &max_element_scalar::compute<scalar_type>, _1, _2 ) );
		return answer;
	}
};

template<typename T>
T max_element( T const &x, T const &y ) {
	return mpl::if_< boost::is_scalar<T>, max_element_scalar, max_element_range >::type::compute( x, y );
}

}} // namespace kml::detail

#endif



