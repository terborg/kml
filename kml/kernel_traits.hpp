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
