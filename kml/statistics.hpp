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

#ifndef STATISTICS_HPP
#define STATISTICS_HPP

#include <boost/iterator/iterator_traits.hpp>
#include <boost/range/begin.hpp>
#include <boost/range/empty.hpp>
#include <boost/range/end.hpp>
#include <boost/range/size.hpp>
#include <boost/range/value_type.hpp>
#include <boost/type_traits.hpp>
#include <cassert>
#include <numeric>


namespace kml {

namespace detail {

// multiplied difference, (x-mu_1)(y-mu_2), used by covariance estimator
// used as binary_op2 in std::inner_product (explains operator())

template<typename T>
class multiply_diff: public std::binary_function<T const&, T const &, T> {
public:
    multiply_diff( T const &m_x = static_cast<T>(0), T const &m_y = static_cast<T>(0) ): mean_x(m_x), mean_y(m_y) {}
    inline T operator()(T const &x, T const &y) const {
        return (x-mean_x) * (y-mean_y);
    }
    T mean_x;
    T mean_y;
};

// squared difference (x-mu)^2, used by variance estimator
// used by accumulate (explains operator())

template<typename T>
class squared_diff: public std::binary_function<T const&, T const &, T> {
public:
    squared_diff( T const &m = static_cast<T>(0) ): mean(m) {}
    inline T operator()(T const &x, T const &y) const {
        T temp = y-mean;
        return (x+temp*temp);
    }
    T mean;
};


} // namespace detail


/*!
The sum of a range
\param x a range of scalar values
\return sum(x_i)
*/

template<typename Range>
typename boost::range_value<Range>::type sum( Range const &x ) {
    typedef typename boost::range_value<Range>::type value_type;
    BOOST_STATIC_ASSERT(boost::is_float<value_type>::value);
    return std::accumulate( boost::begin(x), boost::end(x), static_cast<value_type>(0) );
}

/*!
The minimum of a range
\param x a range of scalar values
\return min(x)
*/

template<typename Range>
typename boost::range_value<Range>::type minimum( Range const &x ) {
    return *std::min_element( boost::begin(x), boost::end(x) );
}

/*!
The maximum of a range
\param x a range of scalar values
\return max(x)
*/

template<typename Range>
typename boost::range_value<Range>::type maximum( Range const &x ) {
    return *std::max_element( boost::begin(x), boost::end(x) );
}

/*!
Unbiased estimator for the population mean
\param x a range of scalar values
\return mean(x) = 1/n * sum(x_i)
*/

template<typename Range>
typename boost::range_value<Range>::type mean( Range const &x ) {
    typedef typename boost::range_value<Range>::type value_type;
    BOOST_STATIC_ASSERT(boost::is_float<value_type>::value);
    if (boost::empty(x))
        return std::numeric_limits<value_type>::quiet_NaN();
    else
        return sum(x) / static_cast<value_type>(boost::size(x));
}


/*!
Unbiased estimator for the population variance
\param x a range of scalar values
\return variance(x) = 1/(n-1) * sum( (x_i-mean(x))^2 )
*/

template<typename Range>
typename boost::range_value<Range>::type variance( Range const &x ) {
    typedef typename boost::range_value<Range>::type value_type;
    BOOST_STATIC_ASSERT(boost::is_float<value_type>::value);
    if (boost::size(x)<2)
        return std::numeric_limits<value_type>::quiet_NaN();
    else
        // could this be done more efficiently?
        return std::accumulate( boost::begin(x), boost::end(x), static_cast<value_type>(0),
                                detail::squared_diff<value_type>(mean(x)) ) /
               static_cast<value_type>(boost::size(x)-1);
}



/*!
Unbiased estimator for the population standard deviation
\param x a range of scalar values
\return sqrt(variance(x))
*/

template<typename Range>
typename boost::range_value<Range>::type standard_deviation( Range const &x ) {
    return std::sqrt( variance(x) );
}


/*!
Unbiased estimator for the population covariance
\param x a range of scalar values
\param y a range of scalar values
\return cov(x,y) = 1/(n-1) * sum( (x_i-mean(x))*(y_i-mean(y)) )
*/

template<typename Range>
typename boost::range_value<Range>::type covariance( Range const &x, Range const &y ) {
    typedef typename boost::range_value<Range>::type value_type;
    BOOST_STATIC_ASSERT(boost::is_float<value_type>::value);
    if ( (boost::size(x) != boost::size(y)) || (boost::size(x)<2) )
        return std::numeric_limits<value_type>::quiet_NaN();
    else
        return std::inner_product( boost::begin(x), boost::end(x), boost::begin(y),
                                   static_cast<value_type>(0),
                                   std::plus<value_type>(),
                                   detail::multiply_diff<value_type>(mean(x), mean(y)) ) /
               static_cast<value_type>(boost::size(x)-1);
}

/*! 
Statistical correlation of two ranges
\param x a range of scalar values
\param y a range of scalar values
\return cor(x,y) = cov(x,y) / ( sd(x)*sd(y) )


\todo Also make it work on two ranges of different kinds, e.g. a ublas::vector and a std::vector

*/

template<typename Range>
typename boost::range_value<Range>::type correlation( Range const &x, Range const &y ) {
    return covariance(x,y) / (standard_deviation(x)*standard_deviation(y));
}


/*!
Mean square, a biased estimator for the population variance. 
It differs from the variance in that it is divided by N, not by N-1.
\param x a range of scalar values
\return mean_square(x) = 1/n * sum( (x_i-mean(x))^2 )
*/

template<typename Range>
typename boost::range_value<Range>::type mean_square( Range const &x ) {
    typedef typename boost::range_value<Range>::type value_type;
    BOOST_STATIC_ASSERT(boost::is_float<value_type>::value);
    if (boost::empty(x))
        return std::numeric_limits<value_type>::quiet_NaN();
    else
        return std::accumulate( boost::begin(x), boost::end(x), static_cast<value_type>(0),
                                detail::squared_diff<value_type>(mean(x)) ) /
               static_cast<value_type>(boost::size(x));
}

/*!
Root-mean-square, a biased estimator for the population standard deviation.
It differs from the standard deviation in that it is divided by N, not by N-1.
\param x a range of scalar values
\return root_mean_square(x) = sqrt(mean_square(x))
*/

template<typename Range>
typename boost::range_value<Range>::type root_mean_square( Range const &x ) {
    return std::sqrt( mean_square( x ) );
}

} // namespace kml

#endif
