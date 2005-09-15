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
#include <boost/range/const_iterator.hpp>
#include <boost/range/size.hpp>
#include <boost/range/value_type.hpp>
#include <boost/type_traits.hpp>
#include <cassert>
#include <cmath>
#include <kml/input_value.hpp>
#include <numeric>

#include <boost/bind.hpp>
#include <kml/detail/prod_element.hpp>
#include <kml/detail/div_element.hpp>
#include <kml/detail/min_element.hpp>
#include <kml/detail/max_element.hpp>
#include <kml/detail/zero_element.hpp>
#include <kml/detail/sqrt_element.hpp>

#include <iostream>


namespace kml {

namespace detail {

// multiplied difference, (x-mu_1)(y-mu_2), used by covariance estimator
// used as binary_op2 in std::inner_product (explains operator())

template<typename T>
class multiply_diff: public std::binary_function<T const&, T const &, T> {
public:
    multiply_diff( T const &m_x, T const &m_y ): mean_x(m_x), mean_y(m_y) {}
    inline T operator()(T const &x, T const &y) const {
        return prod_element(x-mean_x, y-mean_y);
    }
    T mean_x;
    T mean_y;
};


// squared difference (x-mu)^2, used by variance estimator
// used by accumulate (explains operator())

template<typename T>
class squared_diff: public std::binary_function<T const&, T const &, T> {
public:
    squared_diff( T const &m ): mean(m) {}
    inline T operator()(T const &x, T const &y) const {
        T temp = y-mean;
        return (x+prod_element(temp,temp));
    }
    T mean;
};

} // namespace detail


/*!
The sum of a range
\param x a range of either scalars or vectors
\return sum(x_i)

A requirement is that the plus operator is defined on the value_type of the range.
*/

template<typename Range>
typename boost::range_value<Range>::type sum( Range const &x ) {
    typedef typename boost::range_value<Range>::type value_type;
    typedef typename boost::range_const_iterator<Range>::type const_iterator_type;
    if ( boost::size(x) == 1 )
    	return *boost::begin(x);
    else {
	const_iterator_type my_iterator = boost::begin(x);
	++my_iterator;
    	return std::accumulate( my_iterator, boost::end(x), static_cast<value_type>(*boost::begin(x)) );
    }
}

/*!
The minimum of a range
\param x a range of scalar values or vectors
\return min(x)

A requirement is that std::max works on the value_type of the range. If x is a vector, a vector is returned with
element-wise minima.
*/

template<typename Range>
typename boost::range_value<Range>::type minimum( Range const &x ) {
    typedef typename boost::range_value<Range>::type value_type;
    typedef typename boost::range_const_iterator<Range>::type const_iterator_type;
    if ( boost::size(x) == 1 )
    	return *boost::begin(x);
    else {
	const_iterator_type my_iterator = boost::begin(x);
	++my_iterator;
    	return std::accumulate( my_iterator, boost::end(x), static_cast<value_type>(*boost::begin(x)), 
	                        boost::bind( &detail::min_element<value_type>, _1, _2 ) );
    }
}

/*!
The maximum of a range
\param x a range of scalar values or vectors
\return max(x)

A requirement is that std::max works on the value_type of the range. If x is a vector, a vector is returned with
element-wise maxima.
*/

template<typename Range>
typename boost::range_value<Range>::type maximum( Range const &x ) {
    typedef typename boost::range_value<Range>::type value_type;
    typedef typename boost::range_const_iterator<Range>::type const_iterator_type;
    if ( boost::size(x) == 1 )
    	return *boost::begin(x);
    else {
	const_iterator_type my_iterator = boost::begin(x);
	++my_iterator;
    	return std::accumulate( my_iterator, boost::end(x), static_cast<value_type>(*boost::begin(x)), 
				boost::bind( &detail::max_element<value_type>, _1, _2 ) );
    }
}

/*!
Unbiased estimator for the population mean
\param x a range of either scalar values or vectors
\return mean(x) = 1/n * sum(x_i)

A requirement is that the plus operator is defined on the value_type of the range, and that
the value_type can be divided by a scalar_type.
*/

template<typename Range>
typename boost::range_value<Range>::type mean( Range const &x ) {
    typedef typename boost::range_value<Range>::type value_type;
    typedef typename input_value<value_type>::type scalar_type;
    if (boost::empty(x))
        return std::numeric_limits<value_type>::quiet_NaN();
    else
        return sum(x) / static_cast<scalar_type>( boost::size(x) );
}


/*!
Unbiased estimator for the population variance
\param x a range of scalar values or vectors
\return variance(x) = 1/(n-1) * sum( (x_i-mean(x))^2 )

In case of vector values, it will return a vector with the variance for each element
*/

template<typename Range>
typename boost::range_value<Range>::type variance( Range const &x ) {
    typedef typename boost::range_value<Range>::type value_type;
    typedef typename input_value<value_type>::type scalar_type;
    if (boost::size(x)<2)
        return std::numeric_limits<value_type>::quiet_NaN();
    else {
        return std::accumulate( boost::begin(x), boost::end(x), detail::zero_element(*boost::begin(x)),
                                detail::squared_diff<value_type>(mean(x)) ) /
               static_cast<scalar_type>(boost::size(x)-1);
    }
}

/*!
Unbiased estimator for the population standard deviation
\param x a range of scalar values or vectors
\return sqrt(variance(x))

In case of vector values, it will return a vector with the standard deviation for each element
*/

template<typename Range>
typename boost::range_value<Range>::type standard_deviation( Range const &x ) {
    return detail::sqrt_element( variance(x) );
}


/*!
Mean square, a biased estimator for the population variance. 
It differs from the variance in that it is divided by N, not by N-1.
\param x a range of scalar values or vectors
\return mean_square(x) = 1/n * sum( (x_i-mean(x))^2 )

In case of vector values, it will return a vector with the mean square for each element
*/

template<typename Range>
typename boost::range_value<Range>::type mean_square( Range const &x ) {
    typedef typename boost::range_value<Range>::type value_type;
    typedef typename input_value<value_type>::type scalar_type;
    if (boost::empty(x))
        return std::numeric_limits<value_type>::quiet_NaN();
    else
        return std::accumulate( boost::begin(x), boost::end(x), detail::zero_element(*boost::begin(x)),
                                detail::squared_diff<value_type>(mean(x)) ) /
               static_cast<scalar_type>(boost::size(x));
}

/*!
Root-mean-square, a biased estimator for the population standard deviation.
It differs from the standard deviation in that it is divided by N, not by N-1.
\param x a range of scalar values or vectors
\return root_mean_square(x) = sqrt(mean_square(x))

In case of vector values, it will return a vector with the root-mean-square for each element
*/

template<typename Range>
typename boost::range_value<Range>::type root_mean_square( Range const &x ) {
    return detail::sqrt_element( mean_square(x) );
}


/*!
Normalised-mean-square

\param x a range of scalar values or vectors
\return normalised_mean_square(x) = sqrt(mean_square(x))

In case of vector values, it will return a vector with the normalised-mean-square for each element

TODO
*/


template<typename Range>
typename boost::range_value<Range>::type normalised_mean_square( Range const &x ) {
    typedef typename boost::range_value<Range>::type value_type;
    typedef typename input_value<value_type>::type scalar_type;
    if (boost::empty(x))
        return std::numeric_limits<value_type>::quiet_NaN();
    else
        return std::accumulate( boost::begin(x), boost::end(x), detail::zero_element(*boost::begin(x)),
                                detail::squared_diff<value_type>(mean(x)) ) /
               static_cast<scalar_type>(boost::size(x));
}


/*!
Unbiased estimator for the population covariance
\param x a range of scalar values or vectors
\param y a range of scalar values or vectors
\return cov(x,y) = 1/(n-1) * sum( (x_i-mean(x))*(y_i-mean(y)) )

In case of vector values, it will return a vector with the covariance per element -- not a covariance matrix
*/

template<typename Range>
typename boost::range_value<Range>::type covariance( Range const &x, Range const &y ) {
    typedef typename boost::range_value<Range>::type value_type;
    typedef typename input_value<value_type>::type scalar_type;
    if ( (boost::size(x) != boost::size(y)) || (boost::size(x)<2) )
        return std::numeric_limits<value_type>::quiet_NaN();
    else
        return std::inner_product( boost::begin(x), boost::end(x), boost::begin(y),
                                   detail::zero_element(*boost::begin(x)),
                                   std::plus<value_type>(),
                                   detail::multiply_diff<value_type>(mean(x), mean(y)) ) /
               static_cast<scalar_type>(boost::size(x)-1);
}

/*! 
Statistical correlation of two ranges
\param x a range of scalar values or vectors
\param y a range of scalar values or vectors
\return cor(x,y) = cov(x,y) / ( sd(x)*sd(y) )

In case of vector values, it will return a vector with the correlation per element
*/

template<typename Range>
typename boost::range_value<Range>::type correlation( Range const &x, Range const &y ) {
    return detail::div_element( covariance(x,y), detail::prod_element( standard_deviation(x), standard_deviation(y) ) );
}


} // namespace kml

#endif
