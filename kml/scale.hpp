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

#ifndef SCALE_HPP
#define SCALE_HPP

#include <kml/statistics.hpp>
#include <kml/detail/reciprocal_element.hpp>
#include <kml/detail/prod_element.hpp>

namespace kml { namespace detail {

template<typename T>
class linear_transform: public std::unary_function<T const&, T> {
public:
    linear_transform( T const &a, T const &m ): add(a), mult(m) {}
    inline T operator()(T const &x) const {
        return prod_element<T>( mult, x+add );
    }
    T add;
    T mult;
};

} // namespace detail



/*!
Scale a range of data, with mean to mean (default 0) and standard deviation to sd (default 1)
\param x a range of scalars x_i
\pre length of range > 1 (to compute standard deviation)
\pre standard_deviation(range) > machine epsilon (to be able to devide by standard deviation)
\returns x_i = (x_i - mean(x)) / sd(x)

If a standard deviation happens to be (too close to) zero, it is forced to be 1. The mean will 
be subtracted in that case, so that element will be 0 for all samples.
*/

template<typename Range>
void scale_mean_sd( Range &x ) {
    typedef typename boost::range_value<Range>::type value_type;
    // to compute the standard deviation, we need at least two samples
    if (boost::size(x)>1) {
        std::transform( boost::begin(x), boost::end(x), boost::begin(x),
                        detail::linear_transform<value_type>(-mean(x), 
			detail::reciprocal_element(standard_deviation(x))) );
    }
}


/*!
Scale a range of data, with minimum to min (default 0) and maximum to max (default 1)
\param x a range of scalars x_i or vectors x_i
\pre length of range > 1 (to have different minimum and maximum)
\pre standard_deviation(range) > machine epsilon (to be able to devide by difference in max and min)
\returns x_i = (x_i - min(x)) / (max(x)-min(x))
*/

template<typename Range>
void scale_min_max( Range &x ) {
    typedef typename boost::range_value<Range>::type value_type;
    // to get a different a minimum and maximum, we need at least two samples
    if (boost::size(x)>1) {
        value_type min(minimum(x));
        std::transform( boost::begin(x), boost::end(x), boost::begin(x),
                        detail::linear_transform<value_type>(-min, 
                        detail::reciprocal_element<value_type>(maximum(x)-min) ));
    }
}



} // namespace kml


#endif


