
#ifndef BOOST_SERIALIZATION_UBLAS_MATRIX_HPP
#define BOOST_SERIALIZATION_UBLAS_MATRIX_HPP


// copyright Rutger ter Borg, to be under the Boost license
//
// temporary include file for ublas vector type
// acquired from the boost.devel mailing list
// is included here until it is in boost

#include <boost/serialization/nvp.hpp>
#include <boost/serialization/split_free.hpp>
#include <boost/serialization/level.hpp>
#include <boost/serialization/tracking.hpp>


#include <boost/numeric/ublas/matrix.hpp>


namespace ublas = boost::numeric::ublas;


namespace boost {
namespace serialization {


template<class T>
struct tracking_level<ublas::matrix<T> >
{
  typedef mpl::integral_c_tag tag;
  typedef mpl::int_<track_never> type;
  BOOST_STATIC_CONSTANT(
                        int,
                        value = tracking_level::type::value
                        );

}; 
	

template<class T>
struct implementation_level<ublas::matrix<T> >
{
typedef mpl::integral_c_tag tag;
// typedef mpl::int_<primitive_type> type;
typedef mpl::int_<object_serializable> type;
  BOOST_STATIC_CONSTANT( 
           int,
           value = implementation_level::type::value
  );
};


template<class Archive, typename U>
inline void save (Archive &ar, const ublas::matrix<U> &m, const unsigned int) {

  unsigned int count1 = m.size1();
  unsigned int count2 = m.size2();
	
  ar << BOOST_SERIALIZATION_NVP (count1);
  ar << BOOST_SERIALIZATION_NVP (count2);
  
  for ( unsigned int row = 0; row < count1; ++row ) {
	  for( unsigned int col = 0; col < count2; ++col ) {
   	    ar << boost::serialization::make_nvp ("item", m(row,col) );
      }
  }
}
  

template<class Archive, typename U>
inline void load (Archive &ar, ublas::matrix<U> &m, const unsigned int) {

	unsigned int count1;
	unsigned int count2;
    ar >> BOOST_SERIALIZATION_NVP (count1);
    ar >> BOOST_SERIALIZATION_NVP (count2);
    
    m.resize( count1, count2 );
    
   for ( unsigned int row = 0; row < count1; ++row ) {
	  for( unsigned int col = 0; col < count2; ++col ) {
   	    ar >> boost::serialization::make_nvp ("item", m(row,col) );
      }
   }
}


template<class Archive, typename U>
inline void serialize (Archive &ar, ublas::matrix<U>& m, const unsigned int file_version) {
boost::serialization::split_free (ar, m, file_version);
}








} //namespace serialization
} // namespace boost




#endif // BOOST_SERIALIZATION_UBLAS_MATRIX_HPP 
