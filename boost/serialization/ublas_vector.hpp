
#ifndef BOOST_SERIALIZATION_UBLAS_VECTOR_HPP
#define BOOST_SERIALIZATION_UBLAS_VECTOR_HPP


// copyright Neal Becker, presumed to be under the Boost license

// temporary include file for ublas vector type
// acquired from the boost.devel mailing list
// is included here until it is in boost

#include <boost/serialization/nvp.hpp>
#include <boost/serialization/split_free.hpp>
#include <boost/serialization/level.hpp>
#include <boost/serialization/tracking.hpp>


#include <boost/numeric/ublas/vector.hpp>


namespace ublas = boost::numeric::ublas;


namespace boost {
namespace serialization {


template<class T>
struct tracking_level<ublas::vector<T> >
{
  typedef mpl::integral_c_tag tag;
  typedef mpl::int_<track_never> type;
  BOOST_STATIC_CONSTANT(
                        int,
                        value = tracking_level::type::value
                        );

}; 
	

template<class T>
struct implementation_level<ublas::vector<T> >
{
typedef mpl::integral_c_tag tag;
// typedef mpl::int_<primitive_type> type;
typedef mpl::int_<object_serializable> type;
BOOST_STATIC_CONSTANT(
int,
value = implementation_level::type::value
);
};


template<class Archive, class U>
inline void save (Archive &ar, const ublas::vector<U> &v, const unsigned int) {
unsigned int count = v.size();
ar << BOOST_SERIALIZATION_NVP (count);
typename ublas::vector<U>::const_iterator it = v.begin();
while (count-- > 0) {
ar << boost::serialization::make_nvp ("item", *it++);
}
}

template<class Archive, class U>
inline void load (Archive &ar, ublas::vector<U> &v, const unsigned int) {
unsigned int count;
ar >> BOOST_SERIALIZATION_NVP (count);
v.resize (count);
typename ublas::vector<U>::iterator it = v.begin();
while (count-- > 0) {
ar >> boost::serialization::make_nvp ("item", *it++);
}
}


template<class Archive, class U>
inline void serialize (Archive &ar, ublas::vector<U>& v, const unsigned int file_version) {
boost::serialization::split_free (ar, v, file_version);
}




} // namespace serialization
} // namespace boost









#endif


