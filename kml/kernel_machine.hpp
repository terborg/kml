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

#ifndef KERNEL_MACHINE_HPP
#define KERNEL_MACHINE_HPP

#include <boost/lambda/bind.hpp>
#include <boost/utility/enable_if.hpp>
#include <kml/regression.hpp>
#include <kml/classification.hpp>
#include <kml/ranking.hpp>
#include <vector>


#include <boost/serialization/access.hpp>
#include <boost/serialization/tracking.hpp>


//#include <kml/determinate.hpp>

// for the property traits
#include <boost/property_map.hpp>


namespace lambda = boost::lambda;


namespace kml {

template< typename Problem, typename Kernel, typename PropertyMap, class Enable=void >
class kernel_machine {}
;




/*! \brief Regression kernel machine
 
	This is used as a base for various regression kernel machines, for example kml::rvm.
 
	\param Problem a regression problem type, for example kml::regression
	\param Kernel kernel to be used by the machine
 
	\ingroup kernel_machines
*/
template< typename Problem, typename Kernel, typename PropertyMap >
class kernel_machine< Problem, Kernel, PropertyMap, typename boost::enable_if< is_regression<Problem> >::type>:
    public std::unary_function< typename Problem::input_type, typename Problem::output_type > {

public:
    typedef Kernel kernel_type;
    typedef typename Problem::input_type input_type;
    typedef typename Problem::output_type output_type;
    typedef typename boost::property_traits<PropertyMap>::key_type key_type;
    typedef typename boost::property_traits<PropertyMap>::value_type object_type;

    friend class boost::serialization::access;


    // FIXME make this something else...
    typedef double scalar_type;

    // Bug fixed: use call_traits instead of const&, so can be called with both reference and non-reference
    /*! \brief Initializes the kernel
    	\param k parameter used to initialize the kernel
    */
    kernel_machine( typename boost::call_traits<kernel_type>::param_type k,
                    typename boost::call_traits<PropertyMap>::param_type map ):
    kernel_function(k), data(&map) {}


    void set_data( PropertyMap const &map ) {
        data = &map;
    }

    void set_kernel( kernel_type const &k ) {
        kernel_function = k;
    }


    typename kernel_type::return_type kernel( key_type const i, key_type const j ) {
                                    return kernel_function( (*data)[i].get<0>(), (*data)[j].get<0>() );
                                }

                                typename kernel_type::return_type kernel( input_type const &x, key_type const j ) {
                                                                return kernel_function( x, (*data)[j].get<0>() );
                                                            }

                                                            template<typename KeyIterator, typename OutputIterator>
    void fill_kernel( input_type const &x,
                      KeyIterator const begin, KeyIterator const end, OutputIterator out ) {
        KeyIterator i(begin);
        OutputIterator j(out);
        while (i!=end) {
            *j = kernel_function( x, (*data)[*i].get<0>() );
            ++i;
            ++j;
        }
    }

    template<typename KeyIterator, typename OutputIterator>
    void fill_kernel( key_type const key,
                      KeyIterator const begin, KeyIterator const end, OutputIterator out ) {
        KeyIterator i(begin);
        OutputIterator j(out);
        while (i!=end) {
            *j = kernel_function( (*data)[key].get<0>(), (*data)[*i].get<0>() );
            ++i;
            ++j;
        }
    }



    // to fill e.g. a column of H, use fill_kernel
    // from boost documentation:
    // An algorithm that iterates through the range [m.begin1 (), m.end1 ()) will
    // pass through every row of m , an algorithm that iterates through the range [m.begin2 (), m.end2 ())
    // will pass through every column of m .
    template<typename KeyIterator, typename Matrix>
    void design_matrix( KeyIterator const begin, KeyIterator const end, Matrix &out ) {

        // row by row filling
        KeyIterator i(begin);
        std::size_t row = 0;
        while( i != end ) {
            out( row, 0 ) = 1.0;
            std::size_t col = 0;
            KeyIterator j(begin);
            while( j != end ) {
                out( row, ++col ) = kernel_function( (*data)[*i].get<0>(), (*data)[*j].get<0>() );
                ++j;
            }
            ++row;
            ++i;
        }
    }

    // loading and saving capabilities
    template<class Archive>
    void serialize( Archive &archive, unsigned int const version ) {

        // here we store whatever we want to store...
        archive & kernel_function;


    }


    /// Clears the machine (operator() it will always returns 0).
    //     void clear() {
    //         bias = 0.0;
    //         weight.clear();
    //     }

    /// kernel_function used by the machine
    kernel_type kernel_function;

    /// pointer to data container used by the machine
    PropertyMap const *data;


    /// to translate to a sequential view
    //     std::map< key_type, std::size_t > key_mapping;

    typedef typename std::vector< key_type >::size_type index_type;
    //std::vector< key_type > key_lookup;

    /// bias of the machine
    //double bias;
    /// weights of the support vectors
    /// weight[i] is associated with key_lookup[i]
    //std::vector<double> weight;

};





/*! \brief Classification kernel machine
 
	This is used as a base for various classification kernel machines.
 
	\param Problem a classification problem type, for example kml::classification
	\param Kernel kernel to be used by the machine
 
	\ingroup kernel_machines
*/
template< typename Problem, typename Kernel, typename PropertyMap >
class kernel_machine<Problem, Kernel, PropertyMap, typename boost::enable_if< is_classification<Problem> >::type >:
    public std::unary_function< typename Problem::input_type, typename Problem::output_type > {

public:
    typedef Kernel kernel_type;
    typedef typename Problem::input_type input_type;
    typedef typename Problem::output_type output_type;
    typedef typename boost::property_traits<PropertyMap>::key_type key_type;
    typedef typename boost::property_traits<PropertyMap>::value_type object_type;

    // FIXME make this something else...
    typedef double scalar_type;


    kernel_machine( typename boost::call_traits<kernel_type>::param_type k,
                    typename boost::call_traits<PropertyMap>::param_type map ):
    kernel_function(k), data(&map) {}


    typename kernel_type::return_type kernel( key_type const i, key_type const j ) {
                                    return kernel_function( (*data)[i].get<0>(), (*data)[j].get<0>() );
                                }

                                typename kernel_type::return_type kernel( input_type const &x, key_type const j ) {
                                                                return kernel_function( x, (*data)[j].get<0>() );
                                                            }


                                                            template<typename KeyIterator, typename OutputIterator>
    void fill_kernel( input_type const &x,
                      KeyIterator const begin, KeyIterator const end, OutputIterator out ) {
        KeyIterator i(begin);
        OutputIterator j(out);
        while (i!=end) {
            if ( (*data)[*i].get<1>() )
                *j = kernel_function( x, (*data)[*i].get<0>() );
            else
                *j = -kernel_function( x, (*data)[*i].get<0>() );
            ++i;
            ++j;
        }
    }

    template<typename KeyIterator, typename OutputIterator>
    void fill_kernel( key_type const key,
                      KeyIterator const begin, KeyIterator const end, OutputIterator out ) {
        KeyIterator i(begin);
        OutputIterator j(out);
        if ( (*data)[key].get<1>() ) {
            while (i!=end) {
                if ( (*data)[*i].get<1>() )
                    *j = kernel_function( (*data)[key].get<0>(), (*data)[*i].get<0>() );
                else
                    *j = -kernel_function( (*data)[key].get<0>(), (*data)[*i].get<0>() );
                ++i;
                ++j;
            }
        } else {
            while (i!=end) {
                if ( (*data)[*i].get<1>() )
                    *j = -kernel_function( (*data)[key].get<0>(), (*data)[*i].get<0>() );
                else
                    *j = kernel_function( (*data)[key].get<0>(), (*data)[*i].get<0>() );
                ++i;
                ++j;
            }
        }
    }



    // FIXME optimal return type (by value, return value, i.e. see boost call_traits)
    //     output_type operator()( typename boost::call_traits<input_type>::param_type x ) {
    //
    // 	// this must be convertable to bool
    //         return std::inner_product( weight.begin(),
    // 	                           weight.end(),
    // 				   support_vector.begin(),
    // 				   bias,
    //                                    std::plus<output_type>(), boost::lambda::bind(detail::multiplies<double,output_type>(), boost::lambda::_1,
    //                                                              boost::lambda::bind(kernel,x,boost::lambda::_2)) ) >= 0.0;
    //     }

    void set_data( PropertyMap const &map ) {
        data = &map;
    }


    //     void clear() {
    //         bias = 0.0;
    //         weight.clear();
    //     }

    /// kernel_function used by the machine
    kernel_type kernel_function;

    /// pointer to data container used by the machine
    PropertyMap const *data;


    /// to translate to a sequential view
    //     std::map< key_type, std::size_t > key_mapping;
    //     std::vector< key_type > key_lookup;


    //     double bias;

    // the weight should have the sign of the corresponding output sample!
    // w_i = a_i * y_i
    /*    std::vector<double> weight;*/
};



} // namespace kml




// set tracking to track never

namespace boost {
namespace serialization {

template< typename Problem, typename Kernel, typename PropertyMap, typename Enable >
struct tracking_level< kml::kernel_machine<Problem,Kernel,PropertyMap,Enable> > {
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
