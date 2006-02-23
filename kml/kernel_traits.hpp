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

#ifndef KERNEL_TRAITS_HPP
#define KERNEL_TRAITS_HPP

namespace kml { namespace detail {

// generic kernel traits
template<class Kernel>
struct kernel_traits {
	typedef typename Kernel::result_type result_type;
};

} // namespace detail


// traits classes
/*! \brief kernel_result<kernel>::type is kernel::result_type
	\ingroup meta
*/
template<class Kernel>
struct kernel_result {
	typedef typename detail::kernel_traits<Kernel>::result_type type;
};


} // namespace kml

#endif
