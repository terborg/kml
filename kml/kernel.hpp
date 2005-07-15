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

#ifndef KERNEL_HPP
#define KERNEL_HPP


#include <boost/function.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/array.hpp>


/*!

\mainpage The Kernel Machine Library
\author Rutger W. ter Borg
\version 0.1
\section section0 Introduction




\section section2 Concepts

Concepts are sets of requirements on a type, so it can be correctly used as an argument in a call
to a generic algorithm. 

\subsection subsection0 Input and Output types 

Input is either a base type (double, int, etc.), or a Container type (vector, list, sparse vector, etc.).
The requirements of this Container type depend on the kernel to be used on the Input type.

Output is either a float type in case of regression, a binary in case of binary classification, or an integer
in case of multiclass classification.


\section seckern Kernels


\subsection subsection1 Nonmercer Kernel

The generic Kernel concept defines a kernel which computes a pairwise similarity between objects. It can be used 
to define e.g. the similarity between two words, or other input patterns.

- Valid expressions.
- Associated types:      input_type, return_type, parameter_type, ...., etc.
- Semantic invariants.
- Complexity guarantees. A kernel typically at most takes O(N+C) time, with N the size of the input pattern(s) and C
                         some constant.

\code
set_parameter
get_parameter
inner_product( u, v )
\endcode


Models:
- sigmoid

\subsection somesubsect Mercer Kernel

A Mercer Kernel is a Kernel with some additional requirements which have to be fulfilled. A Mercer kernel defines an inner product
in a feature space. In this case, the kernel function is positive definite. 

This allows for additional member functions to be defined:

- dimension(). Returns the dimension of the underlying feature space.
- distance( u, v ). Returns the distance of objects I1 and I2 in the underlying feature space.

Derivatives

Models:
- gaussian
- polynomial

Derivatives.
Derivative object generator.


\subsection subseccache Matrix Cache Policy

Not yet done, but this should select the way all computations are cached and performed.

- Kernel matrix
- Design matrix
- H^T H, i.e. inner product of design matrices
- H^T y

Updates and computation of different types of inverse matrices.

Insert, erase, etc..


\section seckernel Kernel Machines



\subsection submode Determinate Machine


The determinate machine is the well-known defined equation for kernel machines. For regression,
the following expression is assumed

\f$ f(x) = w_0 + \sum_i w_i k(x_i,x) \f$

and for classification, the following



The type of task - Classification or Regression- is determined by C++ type system,
when the outputs are a floating point type, regression is performed, and when the outputs
are a binary or integer type, classification should be performed.


Supported methods:

\code
result_type operator()( Input );               // return the kernel machine at the input

template<class IRange, class ORange>
void learn( Range const &input, Range const &output );

\endcode

Available models:

- svm

\subsection submode2 Probabilistic Machine

A Probabilistic Kernel Machine is a Determinate Machine that additionally assumes some kind of probabilty model on the data.

\code
std::pair<result_type,result_type> operator()( Input );       // compute mean and variance at the input
\endcode


Available models:
- rvm
- srvm
- figueiredo


\subsection asubmode2 Greedy Determinate Machine


\subsection bsubmode2 Greedy Probabilistic Machine


\subsection subsectonline Online Determinate Machine



\subsection subsectonlineprob Online Probabilistic Machine

\subsection subsectgreedy Online Greedy Determinate Machine

Available models:
- krls


\subsection subsectgreedyprob Online Greedy Probabilistic Machine



\section example Examples

\code

// create a kernel
kml::gaussian< ublas::vector<double> > my_kernel( 1.0 );

// compute the kernel on two inputs
ublas::vector<double> u(2);
ublas::vector<double> v(2);

std::cout << my_kernel( u, v ) << std::endl;

// create an online support vector machine
kml::online_svm< ublas::vector<double>, double, kml::gaussian > my_machine( 1.0, 0.1, 10.0 );

// create a vector of 50 input-output pairs
std::vector< ublas::vector<double> > my_inputs( 50 );
std::vector< double > my_outputs( 50 );


// fill the data
for( int i=0; i<50; ++i ) {
	double x = (double(i)/double(49))*20.0-10.0;
	my_inputs[i].resize(1);
	my_inputs[i](0) = x;
	my_outputs[i] = sin(x)/x;
}

// train the whole data set
my_machine.learn( my_inputs, my_outputs );

// produce the predictions
std::vector< double > test( 50 );
std::transform( my_inputs.begin(), my_inputs.end(), test.begin(), my_machine );


\endcode




*/








#endif
