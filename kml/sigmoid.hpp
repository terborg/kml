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

#ifndef SIGMOID_HPP
#define SIGMOID_HPP

#include <boost/call_traits.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/tracking.hpp>
#include <boost/tokenizer.hpp>
#include <boost/type_traits.hpp>
#include <kml/input_value.hpp>
#include <kml/linear.hpp>
#include <cmath>

namespace kml {

/*!
\brief Sigmoid kernel
 
\ingroup kernels
*/

template<typename T>
class sigmoid: public std::binary_function<T, T, typename input_value<T>::type> {
public:
    /*! Refinement of AdaptableBinaryFunction */
    typedef T first_argument_type;
    typedef T second_argument_type;
    typedef typename input_value<T>::type result_type;

    /*! Refinement of Kernel? */
    typedef sigmoid<T> type;

    // scalar type..
    typedef typename input_value<T>::type scalar_type;
    friend class boost::serialization::access;

    /*! Refinement of DefaultConstructible */
    sigmoid(): scale(1.0), bias(0.0) {}

    /*! Refinement of CopyConstructible */
    sigmoid( type const &other ) {
        copy( other );
    }

    /*! Refinement of Assignable */
    type &operator=( type const &other ) {
        if (this != &other) {
            destroy();
            copy(other);
        }
        return *this;
    }

    /*! Construct a sigmoid kernel by providing a values for gamma and lambda
        \param gamma  the scale of the inner product
        \param lambda the bias of the inner product */
    sigmoid( typename boost::call_traits<scalar_type>::param_type gamma,
             typename boost::call_traits<scalar_type>::param_type lambda ): scale(gamma), bias(lambda) {}

    /*! Kernel constructor by providing TokenIterators */
    template<typename Separator>
    sigmoid( typename boost::tokenizer<Separator>::iterator const begin,
             typename boost::tokenizer<Separator>::iterator const end ) {
        scale = 1.0;
        bias = 0.0;
        typename boost::tokenizer<Separator>::iterator iter(begin);
        if ( iter != end )
            scale = boost::lexical_cast<scalar_type>( *iter++ );
        if ( iter != end )
            bias = boost::lexical_cast<scalar_type>( *iter );
    }

    /*! Returns the result of the evaluation of the sigmoid kernel for its arguments
    	\param u input pattern u
        \param v input pattern v
    	\return \f$ tanh( \gamma * u^T v + \lambda) \f$
    */
    scalar_type operator()( T const &u, T const &v ) const {
        return std::tanh( scale * linear<T>()(u,v) + bias );
    }

    // loading and saving capabilities
    template<class Archive>
    void serialize( Archive &archive, unsigned int const version ) {
        archive & scale;
        archive & bias;
    }

    // for debugging purposes
    friend std::ostream& operator<<(std::ostream &os, type const &k) {
        os << "Sigmoid kernel tanh(" << k.scale << "<u,v>+" << k.bias << ")" << std::endl;
        return os;
    }

private:
    void copy( type const &other ) {
        scale = other.scale;
        bias = other.bias;
    }

    scalar_type scale;
    scalar_type bias;

};

} // namespace kml



namespace boost {
namespace serialization {

template<typename T>
struct tracking_level< kml::sigmoid<T> > {
    typedef mpl::integral_c_tag tag;
    typedef mpl::int_<track_never> type;
    BOOST_STATIC_CONSTANT(
        int,
        value = tracking_level::type::value
    );
};

} // namespace serialization
} // namespace boost


#endif

