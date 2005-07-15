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

#ifndef NEWTON_HPP
#define NEWTON_HPP

#include <boost/ref.hpp>
#include <inner_product.hpp>


template<class Point, class UnaryOp1, class UnaryOp2>
class newton: public std::unary_function<Point,Point> {
public:
    newton( UnaryOp1 f, UnaryOp2 Df ): function(boost::cref(f)),
    derivative_function(boost::cref(Df)) {}

    typedef typename boost::range_value<Point>::type scalar_type;
    

    Point operator()(Point const &x0) {

        //std::cout << "finding root for " << x0;

        Point root(x0);
	Point step(x0.size());
	for( int i=0; i<step.size(); ++i ) step[i] = 1.0;
        
	while( (kml::inner_product( step,step) > std::numeric_limits<scalar_type>::epsilon()) ) {
	     // && (std::fabs(derivative_function.get()(root)) > 1e-3) ) {

//               std::cout << "step*step " << kml::inner_product( step,step) << std::endl;
// 	      std::cout << "root:     " << root << std::endl;
//               std::cout << "f(root):  " << function.get()(root) << std::endl;
//               std::cout << "Df(root): " << derivative_function.get()(root) << std::endl;

             
	     step = (1.0 / derivative_function.get()(root)) * function.get()(root);
     	    root -= step;
	    
	    //double func_1 = 1.0 / f(root);
            //double func_2 = derivative_function.get()(root);

            //root -= function.get()(root);
	    


	}
//    	    int qq;
//    	    std::cin >> qq;

	std::cout << " --> root found: " << root << std::endl;
	return root;
    }


    boost::reference_wrapper<UnaryOp1 const> function;
    boost::reference_wrapper<UnaryOp2 const> derivative_function;

};




template<class Point, class UnaryOp1, class UnaryOp2>
class newton_uneven: public std::unary_function<Point,Point> {
public:
    newton_uneven( UnaryOp1 f, UnaryOp2 Df ): function(boost::cref(f)),
    derivative_function(boost::cref(Df)) {}

    typedef typename boost::range_value<Point>::type scalar_type;
    

    Point operator()(Point const &x0) {

        //std::cout << "finding root for " << x0;

        Point root(x0);
	Point step(2);
	step(0) = 1.0;
	step(1) = 1.0;
        
	while( (kml::inner_product( step,step) > std::numeric_limits<scalar_type>::epsilon()) ) {
	     // && (std::fabs(derivative_function.get()(root)) > 1e-3) ) {

//                std::cout << "root:     " << root << std::endl;
//  	      std::cout << "f(root):  " << function.get()(root) << std::endl;
//                std::cout << "Df(root): " << derivative_function.get()(root) << std::endl;

             
	     step = derivative_function.get()(root);
//	      for( int i=0; i<2; ++i) step[i] = 1.0 / step[i];
     	     root -= function.get()(root) * step;

	    //double func_1 = 1.0 / f(root);
            //double func_2 = derivative_function.get()(root);

            //root -= function.get()(root);
	    


	}

//    	    int qq;
//     	    std::cin >> qq;

	
	std::cout << " --> root found: " << root << std::endl;
	return root;
    }


    boost::reference_wrapper<UnaryOp1 const> function;
    boost::reference_wrapper<UnaryOp2 const> derivative_function;

};













#endif

