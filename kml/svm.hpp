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

#ifndef SVM_HPP
#define SVM_HPP

#include <traits.hpp>
#include <algorithm/aosvr.hpp>
#include <determinate.hpp>



namespace kml {



template< typename I,
	  typename O,
          template<typename,int> class K >
class svm: public determinate<I,O,K> {
public:
	typedef typename determinate<I,O,K>::kernel_type kernel_type;
//	typedef typename determinate<I,O,K> base_type;


	// Bug fixed: use call_traits instead of const&, so can be called with both reference and non-reference
    	svm( typename boost::call_traits<kernel_type>::param_type k,
	     typename boost::call_traits<double>::param_type param_eps,
	     typename boost::call_traits<double>::param_type param_C ): 
	         determinate<I,O,K>(k), epsilon(param_eps), C(param_C) {}

	template< class IRange, class ORange >
	void learn( IRange const &input, ORange const &output ) {
		// use template pattern...
		
		// create a temporary online learner, train SVM
		// and copy results.
		
		//aosvr< svm<I,O,K> > learner;
/*		for( int i=0; i<boost::size(input); ++i ) {
			aosvr< svm<I,O,K> >::push_back( input, output );
			//learner.push_back( input, output );
		}*/
		
//		aosvr< svm<I,O,K> >::learn( input, output );
	}

	double epsilon;
	double C;

};
















}

#endif
