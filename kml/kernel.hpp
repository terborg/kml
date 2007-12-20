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

#ifndef KML_KERNEL_HPP
#define KML_KERNEL_HPP


namespace kml {

template< typename Input, typename Result, typename Super >
class kernel: public std::binary_function<Input, Input, Result > {

    /*! Refinement of AdaptableBinaryFunction */
    typedef Input first_argument_type;
    typedef Input second_argument_type;
    typedef Result result_type;

    inline
    Result operator()( Input const &u, Input const &v ) const {
        return Super::dot( u, v );
    }

    inline
    Result dot_self( Input const &u ) {
    }








};



}

#endif


