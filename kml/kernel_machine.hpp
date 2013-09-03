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
#include <boost/call_traits.hpp>
#include <boost/shared_ptr.hpp>
#include <kml/regression.hpp>
#include <kml/classification.hpp>
#include <kml/ranking.hpp>
#include <vector>

#include <boost/serialization/access.hpp>
#include <boost/serialization/tracking.hpp>

// for the property traits
#include <boost/property_map/property_map.hpp>
#include <boost/ref.hpp>

//using boost::tuples::get;

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

    typedef typename std::vector< key_type >::size_type index_type;

    // FIXME make this something else...
    typedef double scalar_type;

    // Bug fixed: use call_traits instead of const&, so can be called with both reference and non-reference
    /*! \brief Initializes the kernel
    	\param k parameter used to initialize the kernel
    */
    kernel_machine( typename boost::call_traits<kernel_type>::param_type k,
                    PropertyMap const &map ): kernel_function(k), data(boost::cref(map)) {}

//     void set_data( PropertyMap const& map ) {
//         data = map;
//     }

    void set_kernel( kernel_type const &k ) {
        kernel_function = k;
    }

    inline
    input_type const& input( key_type const key ) const {
	return boost::tuples::get<0>(data.get()[key]);
    }

    inline
    output_type const& output( key_type const key ) const {
	return boost::tuples::get<1>(data.get()[key]);
    }


    typename kernel_type::result_type kernel( key_type const i, key_type const j ) {
        return kernel_function( input(i), input(j) );
    }

    typename kernel_type::result_type kernel( input_type const &x1, key_type const j ) {
        return kernel_function( x1, input(j) );
    }

    typename kernel_type::result_type kernel( input_type const &x1, input_type const &x2 ) {
        return kernel_function( x1, x2 );
    }
 

    template<typename KeyIterator, typename OutputIterator>
    void fill_kernel( input_type const &x,
                      KeyIterator const begin, KeyIterator const end, OutputIterator out ) {
        KeyIterator i(begin);
        OutputIterator j(out);
        while (i!=end) {
            *j = kernel_function( x, input(*i) );
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
            *j = kernel_function( input(key), input(*i) );
            ++i;
            ++j;
        }
    }

    template<typename KeyIterator, typename Matrix>
    void kernel_matrix( KeyIterator const begin, KeyIterator const end, Matrix &out ) {
        // row by row filling
        KeyIterator i(begin);
        std::size_t row = 0;
        while( i != end ) {
            std::size_t col = 0;
            KeyIterator j(begin);
            while( j != end ) {
                out( row, col ) = kernel_function( input(*i), input(*j) );
		++col;
                ++j;
            }
            ++row;
            ++i;
        }
    }

    // to fill, e.g., a column of H, use the fill_kernel method
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
                out( row, ++col ) = kernel_function( input(*i), input(*j) );
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



private:
    /// kernel_function used by the machine
    kernel_type kernel_function;

    /// pointer to data container used by the machine
    boost::reference_wrapper< PropertyMap const > data;
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
                    PropertyMap const &map ):
    kernel_function(k), data(boost::cref(map)) {}

//     kernel_machine( typename boost::call_traits<kernel_type>::param_type k,
// 		    PropertyMap const &map ):
// 	kernel_function(k), data(boostmap) { }

    kernel_machine(kernel_machine &k): kernel_function(k.kernel_function),
				       data(k.data) { }


    inline
    input_type const& input( key_type const key ) const {
	return boost::tuples::get<0>(data.get()[key]);
    }

    inline
    output_type const& output( key_type const key ) const {
	return boost::tuples::get<1>(data.get()[key]);
    }

    typename kernel_type::result_type kernel( key_type const i, key_type const j ) {
        return kernel_function( input(i), input(j) );
    }

    typename kernel_type::result_type kernel( input_type const &x1, key_type const j ) {
        return kernel_function( x1, input(j) );
    }

    typename kernel_type::result_type kernel( input_type const &x1, input_type const &x2 ) {
        return kernel_function( x1, x2 );
    }

    template<typename KeyIterator, typename OutputIterator>
    void fill_kernel( input_type const &x,
                      KeyIterator const begin, KeyIterator const end, OutputIterator out ) {
        KeyIterator i(begin);
        OutputIterator j(out);
        while (i!=end) {
            if ( data.get()[*i].get<1>() )
                *j = kernel_function( x, input(*i) );
            else
                *j = -kernel_function( x, input(*i) );
            ++i;
            ++j;
        }
    }

    template<typename KeyIterator, typename OutputIterator>
    void fill_kernel( key_type const key,
                      KeyIterator const begin, KeyIterator const end, OutputIterator out ) {
        KeyIterator i(begin);
        OutputIterator j(out);
        if ( data.get()[key].get<1>() ) {
            while (i!=end) {
                if ( data.get()[*i].get<1>() )
                    *j = kernel_function( input(key), input(*i) );
                else
                    *j = -kernel_function( input(key), input(*i) );
                ++i;
                ++j;
            }
        } else {
            while (i!=end) {
                if ( data.get()[*i].get<1>() )
                    *j = -kernel_function( input(key), input(*i) );
                else
                    *j = kernel_function( input(key), input(*i) );
                ++i;
                ++j;
            }
        }
    }

    void set_kernel( kernel_type const &k ) {
        kernel_function = k;
    }


    // loading and saving capabilities
    template<class Archive>
    void serialize( Archive &archive, unsigned int const version ) {

        // here we store whatever we want to store...
        archive & kernel_function;


    }


private:
    /// kernel_function used by the machine
    kernel_type kernel_function;

    /// pointer to data container used by the machine
    boost::reference_wrapper< PropertyMap const > data;

};

/*! \brief Ranking kernel machine
 
	This is used as a base for various ranking kernel machines, for example kml::svm.
 
	\param Problem a ranking problem type, for example kml::ranking
	\param Kernel kernel to be used by the machine
 
	\ingroup kernel_machines
*/
template< typename Problem, typename Kernel, typename PropertyMap >
class kernel_machine<Problem, Kernel, PropertyMap, typename boost::enable_if< is_ranking<Problem> >::type >:
    public std::unary_function< typename Problem::input_type, typename Problem::output_type > {

public:
    typedef Kernel kernel_type;
    typedef typename Problem::input_type input_type;
    typedef typename Problem::group_type group_type;
    typedef typename Problem::output_type output_type;
    typedef typename boost::property_traits<PropertyMap>::key_type key_type;
    typedef typename boost::property_traits<PropertyMap>::value_type object_type;

    // FIXME make this something else...
    typedef double scalar_type;


    kernel_machine( typename boost::call_traits<kernel_type>::param_type k,
                    typename boost::call_traits<PropertyMap>::param_type map ):
    kernel_function(k), data(&map) {}

    kernel_machine( typename boost::call_traits<kernel_type>::param_type k,
		    PropertyMap const &map ) :
	kernel_function(k), data(boost::cref(map)) {}

    typename kernel_type::result_type kernel( key_type const i, key_type const j ) {
                                    return kernel_function( (*data)[i].get<0>(), (*data)[j].get<0>() );
                                }

                                typename kernel_type::result_type kernel( input_type const &x, key_type const j ) {
                                                                return kernel_function( x, (*data)[j].get<0>() );
                                                            }
      
  typename kernel_type::result_type kernel(key_type const i, input_type const &x) {
    return kernel_function(data.get()[i].get<0>(), x);
  }

                                                            template<typename KeyIterator, typename OutputIterator>
    void fill_kernel( input_type const &x,
                      KeyIterator const begin, KeyIterator const end, OutputIterator out ) {
        KeyIterator i(begin);
        OutputIterator j(out);
        while (i!=end) {
            if ( data.get()[*i].get<1>() )
                *j = kernel_function( x, data.get()[*i].get<0>() );
            else
                *j = -kernel_function( x, data.get()[*i].get<0>() );
            ++i;
            ++j;
        }
    }

    template<typename KeyIterator, typename OutputIterator>
    void fill_kernel( key_type const key,
                      KeyIterator const begin, KeyIterator const end, OutputIterator out ) {
        KeyIterator i(begin);
        OutputIterator j(out);
        if ( data.get()[key].get<1>() ) {
            while (i!=end) {
                if ( data.get()[*i].get<1>() )
                    *j = kernel_function( data.get()[key].get<0>(), data.get()[*i].get<0>() );
                else
                    *j = -kernel_function( data.get()[key].get<0>(), data.get()[*i].get<0>() );
                ++i;
                ++j;
            }
        } else {
            while (i!=end) {
                if ( data.get()[*i].get<1>() )
                    *j = -kernel_function( data.get()[key].get<0>(), data.get()[*i].get<0>() );
                else
                    *j = kernel_function( data.get()[key].get<0>(), data.get()[*i].get<0>() );
                ++i;
                ++j;
            }
        }
    }

    void set_data( PropertyMap const &map ) {
	data = boost::shared_ptr<PropertyMap const>(&map);
    }

    void set_data( PropertyMap const *map) {
	data = boost::shared_ptr<PropertyMap const>(map);
    }

    void set_data( boost::shared_ptr<PropertyMap const> map) {
	data = map;
    }

    // loading and saving capabilities
    template<class Archive>
    void serialize( Archive &archive, unsigned int const version ) {

        // here we store whatever we want to store...
        archive & kernel_function;


    }

    /// kernel_function used by the machine
    kernel_type kernel_function;

    /// pointer to data container used by the machine
    boost::reference_wrapper< PropertyMap const > data;

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
