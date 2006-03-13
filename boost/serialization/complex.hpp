
#ifndef BOOST_SERIALIZATION_COMPLEX_HPP
#define BOOST_SERIALIZATION_COMPLEX_HPP


// copyright Neal Becker, presumed to be under the Boost license

// temporary include file for ublas vector type
// acquired from the boost.devel mailing list
// is included here until it is in boost

#include <boost/serialization/nvp.hpp>
#include <complex>

#include <boost/serialization/level.hpp>
#include <boost/serialization/tracking.hpp>




template<class T>
struct implementation_level<std::complex<T> > {
    typedef mpl::integral_c_tag tag;
    // typedef mpl::int_<primitive_type> type;
    typedef mpl::int_<object_serializable> type;
    BOOST_STATIC_CONSTANT(
        int,
        value = implementation_level::type::value
    );
};

template<class T>
struct tracking_level<std::complex<T> > {
    typedef mpl::integral_c_tag tag;
    typedef mpl::int_<track_never> type;
    BOOST_STATIC_CONSTANT(
        int,
        value = tracking_level::type::value
    );

};


template<class Archive, class T>
inline void serialize (Archive &ar, std::complex<T>& z, const
                       unsigned int file_version) {
    ar & boost::serialization::make_nvp ("real", real(z));
    ar & boost::serialization::make_nvp ("imag", imag(z));
    // ar & real(z);
    // ar & imag(z);
}
}


#endif
