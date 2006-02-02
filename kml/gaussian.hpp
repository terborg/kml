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

#ifndef GAUSSIAN_HPP
#define GAUSSIAN_HPP

#include <boost/call_traits.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/type_traits.hpp>
#include <kml/input_value.hpp>
#include <kml/distance.hpp>
#include <cassert>

namespace kml {
	
/*!
\brief Gaussian kernel of the form \f$ f(x)=e^{-\gamma\|x - \mu\|} \f$ where
	\f$ \gamma = \frac{1}{\sigma^2} \f$
\param I defines the underlying input data type
	(distance_square<I, I> must be defined, which is done for
	scalar and vector types in distance.hpp)

This is a template class that creates a function for the Gaussian kernel. 

\todo
- clean-up
- finish documentation
- complexity guarantees
- loading and saving

\ingroup kernels
*/

template<typename T>
class gaussian: public std::binary_function<T, T, typename input_value<T>::type> {
public:
    BOOST_SERIALIZATION_SPLIT_MEMBER()
    typedef gaussian<T> type;
    friend class boost::serialization::access;
    typedef typename input_value<T>::type scalar_type;
    typedef typename mpl::int_<0>::type derivative_order;
    typedef typename input_value<T>::type return_type;


    /*! Construct an uninitialised Gaussian kernel */
    gaussian() {}

    /*! Refinement of CopyConstructable */
    gaussian( type const &other ) {
       copy( other );
    }

    /*! Refinement of Assignable */
    type operator=( type const &other ) {
	if (this != &other) { destroy(); copy(other); }
	return (*this);
    }

    /*! Construct a Gaussian kernel by providing a value for \f$ \sigma \f$ */
	gaussian( typename boost::call_traits<scalar_type>::param_type sigma ) {
       set_width(sigma);
    }

    /*! Returns the result of the evaluation of the Gaussian kernel for these points
		\param u input pattern u
        \param v input pattern v
	\return \f$ e^{-\gamma\|u - v\|} \f$
    */
    scalar_type operator()( T const &u, T const &v ) const {
        return std::exp( exp_factor * distance_square( u, v ) );
    }

	/*! Sets \f$ \sigma \f$. */
    void set_width( typename boost::call_traits<scalar_type>::param_type sigma ) {
    	assert( sigma > 0.0 );
	width = sigma;
	exp_factor = -1.0 / (2.0*sigma*sigma);
    }
    
	/*! Returns \f$ \sigma \f$. */
	scalar_type const get_width() const {
    	return width;
    }
    
   	/*! Sets \f$ \gamma \f$. */
	void set_gamma( typename boost::call_traits<scalar_type>::param_type gamma ) {
	set_scale_factor( gamma );
    }

    /*! Sets \f$ \gamma \f$. */
	void set_scale_factor( typename boost::call_traits<scalar_type>::param_type gamma ) {
        assert( gamma > 0.0 );
	width = std::sqrt(0.5 / gamma);
	exp_factor = -gamma;
    }

    /*! Returns \f$ \gamma \f$. */
    scalar_type const get_gamma() const {
    	return -exp_factor;
    }
    
    /*! Returns \f$ \gamma \f$. */
    scalar_type const get_scale_factor() const {
    	return -exp_factor;
    }
    
    /*! The dimension of the feature space */
    scalar_type const dimension() const {
	return std::numeric_limits<scalar_type>::infinity();
    }

    template<class Archive>
    void load( Archive &archive, unsigned int const version ) {
    	archive & width;
	exp_factor = -1.0 / (2.0*width*width);
    }
    
    template<class Archive>
    void save( Archive &archive, unsigned int const version ) const {
    	archive & width;
    }
    
    friend std::ostream& operator<<(std::ostream &os, type const &k) {
	os << "Gaussian kernel (width " << k.width << ")" << std::endl;
	return os;
    }

private:
    void copy( type const &other ) {
	width = other.width;
	exp_factor = other.exp_factor;
    }
    void destroy() {}

    scalar_type width;
    scalar_type exp_factor;
};


} // namespace kml


#endif

