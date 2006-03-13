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
 
#ifndef MATH_HPP
#define MATH_HPP

//
// discrete math, compile-time
// combinatoric math
//



namespace kml { namespace detail {

//
// return the sign of an underlying float type
//

template<typename T>
inline const int sign( T const &value ) {
  return( value >= static_cast<T>(0) ? 1 : -1 );
}

//
// checks whether a number should be considered zero, or not,
// i.e. below the machine precision
//

template<typename T>
inline const bool is_zero( T const &value ) {
   return ( (value < std::numeric_limits<T>::epsilon()) &&
            (value > -std::numeric_limits<T>::epsilon()) );
}



//
// static factorial (up to 19 now, ie 32 unsigned bits)
//

template<unsigned int N>
struct factorial {
    static const unsigned int value = N * factorial<N-1>::value;
};

template<>
struct factorial<0> {
    static const unsigned int value = 1;
};

// a run-time counter-part as well...
inline unsigned int factorial_rt( unsigned int n ) {
    return ( n==0 ? 1 : n * factorial_rt(n-1) );
}

//
// "N choose K" (Dutch: N boven K)
// the binomial coefficient (N K)
//

template<unsigned int N, unsigned int K>
class choose {
public:
  static const unsigned int value = factorial<N>::value / (factorial<N-K>::value * factorial<K>::value);
};


//
// static a^b
//


template<int A, unsigned int B>
class power {
public:
    static const int value = A * power<A,B-1>::value;
};

template<int A>
class power<A,0> {
public:
    static const int value = 1;
};

template<unsigned int B>
class power<-1,B> {
public:
    static const int value = (B % 2 ? -1 : 1);
};

template<unsigned int B>
class power<0,B> {
public:
    static const int value = 0;
};

template<unsigned int B>
class power<1,B> {
public:
    static const int value = 1;
};

template<unsigned int B>
class power<2,B> {
public:
    static const int value = 1 << B;
};


template<>
class power<2,0> {
public:
    static const int value = 1;
};

template<>
class power<-1,0> {
public:
    static const int value = 1;
};


//
// static Eulerian number <n,k>
//

template<unsigned int N, unsigned int K, unsigned int J>
struct eulerian_sum {
    static const int value = power<-1,J>::value * choose<N+1,J>::value * power<K-J+1,N>::value +
                             eulerian_sum<N,K,J-1>::value;
};

template<unsigned int N, unsigned int K>
struct eulerian_sum<N,K,0> {
    static const int value = choose<N+1,0>::value * power<K+1,N>::value;
};

template<unsigned int N, unsigned int K>
struct eulerian {
	static const int value = eulerian_sum<N,K,K+1>::value;
};


//
// Entringer number
// http://mathworld.wolfram.com/EntringerNumber.html
//
// e.g. E04..E44 = 0 2 4 5 5

template<unsigned int N, unsigned int K>
struct entringer {
      static const unsigned value = entringer<N,K-1>::value + entringer<N-1,N-K>::value;
};

template<unsigned int N>
struct entringer<N,0> {
      static const unsigned value = (N==0 ? 1 : 0);
};













} // namespace detail


} // namespace kml






#endif
