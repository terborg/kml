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

#ifndef SCALE_OR_DOT_HPP
#define SCALE_OR_DOT_HPP

#include <boost/range/begin.hpp>
#include <boost/range/end.hpp>
#include <boost/range/size.hpp>
#include <boost/range/value_type.hpp>
#include <boost/mpl/eval_if.hpp>
#include <boost/mpl/or.hpp>
#include <boost/mpl/and.hpp>
#include <boost/mpl/identity.hpp>
#include <boost/type_traits/is_scalar.hpp>
#include <boost/type_traits/is_same.hpp>

#include <numeric>
#include <iostream>

namespace mpl = boost::mpl;

namespace kml { namespace detail {

// TypeX   TypeY   Result  Handled by
// scalar  scalar  scalar  scale_product
// vector  scalar  vector  scale_product
// scalar  vector  vector  scale_product
// vector  vector  scalar  dot_product

template<typename TypeX, typename TypeY>
struct scale_or_dot_value: mpl::eval_if< mpl::or_< boost::is_scalar<TypeX>, boost::is_scalar<TypeY> >, 
                   mpl::eval_if< boost::is_same<TypeX,TypeY>, mpl::identity<TypeX>, mpl::if_< boost::is_scalar<TypeX>, TypeY, TypeX > >,
                   boost::range_value<TypeX> > {};

struct scale_product {
	template<typename TypeX, typename TypeY>
	static typename scale_or_dot_value<TypeX,TypeY>::type compute( TypeX const &x, TypeY const &y ) {
		return x * y;
	}
};

struct dot_product {
	template<typename TypeX, typename TypeY>
	static typename scale_or_dot_value<TypeX,TypeY>::type compute( TypeX const &x, TypeY const &y ) {
		typedef typename boost::range_value<TypeX>::type value_type;
		// TODO pass on to a inner-prod specialisation
		return std::inner_product( boost::begin(x), boost::end(x), boost::begin(y), static_cast<value_type>(0) );
	}
};

template<typename TypeX, typename TypeY>
typename scale_or_dot_value<TypeX,TypeY>::type scale_or_dot( TypeX const &x, TypeY const &y ) {
	// figure out what to do (during compile-time)
	return mpl::if_< mpl::or_< boost::is_scalar<TypeX>, boost::is_scalar<TypeY> >, scale_product, dot_product >::type::compute( x, y );
}

}} // namespace kml::detail

#endif



