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

#ifndef ONLINE_SVM_HPP
#define ONLINE_SVM_HPP

#include <boost/numeric/bindings/traits/std_vector.hpp>
#include <boost/numeric/bindings/traits/ublas_matrix.hpp>
#include <boost/numeric/bindings/traits/ublas_symmetric.hpp>
#include <boost/numeric/bindings/traits/ublas_vector.hpp>
#include <boost/mpl/equal_to.hpp>
#include <boost/type_traits/is_float.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/utility/enable_if.hpp>

#include <kml/matrix_view.hpp>
#include <kml/symmetric_view.hpp>

#include <kml/regression.hpp>
#include <kml/classification.hpp>
#include <kml/ranking.hpp>

#include <kml/kernel_machine.hpp>

#include <boost/numeric/ublas/io.hpp>



namespace atlas = boost::numeric::bindings::atlas;
namespace mpl = boost::mpl;

namespace kml {

/*!
\brief On-line Support Vector Machine.
\param Problem the problem type
\param Kernel the kernel type

Training a Support Vector Machine (SVM) normally requires solving a quadratic programming problem (QP) in a number of
coefficients equal to the number of training examples. For very large data sets, QP methods become infeasible. 
Algorithms are available to do incremental and decremental support vector learning, for both classiciation and regression.

Abilities: Incremental, decremental and update of a data set.

See online determinate class description for specifics about memory requirements.
 
Example code:
\code
kml::gaussian< ublas::vector<double> > my_kernel( 1.6 );
kml::online_svm< ublas::vector<double>, double, kml::gaussian > my_machine( my_kernel, 0.1, 10 );
my_machine.train( my_input, my_output );
\endcode
 

In general, we have training input vectors x. In binary SVM classification, 
the optimal seperating function is given by 
\f$ f(x)=\sum_j \alpha_j y_j k(x_j,x) + b \f$ with labels \f$ y_i=\pm 1 \f$. In SVM regression,
the function is similar, \f$ f(x)=\sum_j \alpha_j k(x_j,x) + b \f$. 

One of the differences between classification and regression algorithms is the composition of the
matrix Q and R as referred to in [1-3], and hence the \e coefficient \e sensitivities and the  
\e margin \e sensitivities differ.

Regression: \f$ Q_{ij}=k(x_i,x_j) \f$
Classification: \f$ Q_{ij}= y_i y_j k(x_i,x_j) \f$



\bug
- Fix stability of the regression algorithm against noiseless data. At the moment, this is a largest instability issue.
 
\todo
- small:  introduce a margin-star set, this prevents a test in the algorithm
- fix documentation
- removal of points from the actual set (decremental algorithm(s))
- updates of points in the actual set (update, use a different stopping criterion)


\section bibliography References

-# Junshui Ma et al., 2003. Accurate On-Line Support Vector Regression.
   \e Neural \e Computation, pp. 2700-2701.
   http://tinyurl.com/5g2bb
-# Mario Martin, 2002. On-line Support Vector Machine for Function Approximation.
   Technical report LSI-02-11-R, Universitat Politecnica de Catalunya
   http://tinyurl.com/48y9n
  
\ingroup kernel_machines
*/


template< typename Problem, typename Kernel, typename PropertyMap, class Enable = void>
class online_svm: public kernel_machine<PropertyMap,Problem,Kernel> {};




//
//
// REGRESSION ALGORITHM
//
//

template< typename Problem, typename Kernel, typename PropertyMap >
class online_svm< Problem, Kernel, PropertyMap, typename boost::enable_if< is_regression<Problem> >::type>:
         public kernel_machine< Problem, Kernel, PropertyMap > {
public:
    typedef kernel_machine< Problem, Kernel, PropertyMap > base_type;
    typedef typename base_type::kernel_type kernel_type;
    typedef typename base_type::result_type result_type;
    typedef typename Problem::input_type input_type;
    typedef typename Problem::output_type output_type;
    typedef double scalar_type;
    typedef ublas::symmetric_matrix<double> symmetric_type;
    typedef ublas::matrix<double> matrix_type;
    typedef ublas::vector<double> vector_type;


	typedef typename boost::property_traits<PropertyMap>::key_type key_type;
	typedef typename boost::property_traits<PropertyMap>::value_type object_type;



    // Bug fixed: use call_traits instead of const&, so can be called with both reference and non-reference
    /*! \param k the kernel construction parameter
        \param tube_width the error-insensitive tube width, parameter \f$ \epsilon \f$ of Vapnik's e-insensitive loss function
	\param max_weight the maximum weight assigned to any support vector, support vector machine's parameter C
    */
    online_svm( typename boost::call_traits<scalar_type>::param_type max_weight,
                typename boost::call_traits<scalar_type>::param_type tube_width,
		typename boost::call_traits<kernel_type>::param_type k,
                typename boost::call_traits<PropertyMap>::param_type map ):
    		base_type(k,map), epsilon(tube_width), C(max_weight) {}


    template< typename TokenIterator >
    online_svm( TokenIterator const begin, TokenIterator const end, 
                typename boost::call_traits<kernel_type>::param_type k,
		typename boost::call_traits<PropertyMap>::param_type map ):
		base_type(k,map) {
		C = 10.0;
		epsilon = 0.1;
		TokenIterator token( begin );
		if ( token != end ) {
			C = boost::lexical_cast<double>( *token++ );
			if ( token != end ) epsilon = boost::lexical_cast<double>( *token );
		}
	}


    result_type operator()( input_type const &x ) {
        result_type result = base_type::bias;
        // temp_K[i], not temp_K[i+1]
        if (!margin_set.empty()) {
            vector_type temp_K( margin_set.size() );
    	    base_type::fill_kernel( x, margin_key.begin(), margin_key.end(), temp_K.begin() );
            result += atlas::dot( temp_K, base_type::weight );
        }
        if (!error_set.empty()) {
            result_type temp_K(0);
            //vector_type temp_K( error_set.size() );
            for( unsigned int i=0; i < error_set.size(); ++i )
                temp_K += base_type::kernel( x, base_type::key_lookup[error_set[i]] );
            result += C * temp_K;
        }
        if (!error_star_set.empty()) {
            result_type temp_K(0);
            for( unsigned int i=0; i < error_star_set.size(); ++i )
                temp_K += base_type::kernel( x, base_type::key_lookup[error_star_set[i]] );
            result -= C * temp_K;
        }
        return result;
    }


   /*! learn the entire range of keys indicated by this range */
    template<typename KeyIterator>
    void learn( KeyIterator begin, KeyIterator end ) {
	
	// batch learning algorithm(s) right here...
	KeyIterator key_iterator(begin);
	while( key_iterator != end ) {
		increment( *key_iterator );
		++key_iterator;
	}
    }


    /*! call the incremental algorithm */
    void increment( key_type const key ) {

        // index: * the corresponding row in the design matrix
        //        * the corresponding location in the margin sense vector
        // key:   * the corresponding location in the data container

	std::size_t index = base_type::key_lookup.size();
	base_type::key_lookup.push_back( key );

        // record prediction error
        if (debug) {
            std::cout << "Starting AOSVR incremental algorithm for key " << key << std::endl;
	    std::cout << "              the associated index is        " << index << std::endl;
	}

        // important: the sign of the residual is used below
	residual.push_back( (*base_type::data)[key].get<1>() - operator()( (*base_type::data)[key].get<0>() ) );

        // add to support vector buffer
        //     preserved_resize( all_vectors, index+1, x_t.size() );
        //     row( all_vectors, index ).assign( x_t );
        //     preserved_resize( h, index+1 );


        if ( index == 0 ) {

            // first point, initialise machine
            if (debug)
                std::cout << "Initialising the Accurate Online Support Vector Regression machine" << std::endl;
            residual.back() = 0.0;
            base_type::bias = (*base_type::data)[key].get<1>();
            
	    // put this first point (with index 0) in the remaining set
	    remaining_set.push_back( index );

            // initialise the inverse matrix with a zero only
	    R.grow_row_column();
            R.matrix(0,0) = 0.0;

            // initialise design matrix with bias column
            H.grow_row_column();
            
	    // in case of regression: initialise design matrix with bias column	    
	    H.matrix(0,0) = 1.0;

	    // in case of classification: 
	    
        } else {


            // resize (view in) design matrix
            H.grow_row();

            if (debug)
                std::cout << "H is now " << H.size1() << " by " << H.size2() << std::endl;

	


            // fill last row of design matrix with this input sample
	    // TODO this should be an otimised function call 

	    ublas::matrix_row<ublas::matrix<double> >::iterator j = ublas::row( H.matrix, index ).begin(); 
	    *j++ = 1.0;
	    base_type::fill_kernel( key, margin_key.begin(), margin_key.end(), j );

            if ( std::fabs(residual.back()) <= epsilon ) {
                if (debug)
                    std::cout << "Adding index " << index << " to remaining set." << std::endl;
                remaining_set.push_back( index );

            } else {
                if (debug)
                    std::cout << "re-establishing KKT conditions..." << std::endl;

                if ( residual[ index ] < 0.0 )
                    satisfy_KKT_conditions< incremental_type, mpl::int_<-1> >( index, key, 0.0 );
                else
                    satisfy_KKT_conditions< incremental_type, mpl::int_<1> >( index, key, 0.0 );
            }


        }
    }

    // define types for different actions
    struct incremental_type {
        typedef mpl::bool_<true> increment;
        typedef mpl::bool_<false> decrement;
    };

    struct decremental_type {
        typedef mpl::bool_<false> increment;
        typedef mpl::bool_<true> decrement;
    };




    // this template function switches the direction of a compare operation 
    // on the basis of the direction type
    template<typename direction>
    inline
    bool switch_compare( scalar_type const a, scalar_type const b ) {
        if ( mpl::equal_to<direction,mpl::int_<1> >::type::value )
            return a < b;
        else
            return a > b;
    }

    
    // this template function switches the operation done on two scalars
    // on the basis of the direction type
    template<typename direction>
    inline
    scalar_type switch_add_sub( scalar_type const a, scalar_type const b ) {
        if ( mpl::equal_to<direction,mpl::int_<1> >::type::value )
            return a - b;
        else
            return a + b;
    }

    // scalar_type qm = detail::sign(q*margin_sense(idx));
    // + qm * epsilon
    template<typename direction>
    inline
    scalar_type inline_qm( scalar_type const ms, scalar_type const b ) {
        if ( mpl::equal_to<direction,mpl::int_<1> >::type::value )
            return ( ms < static_cast<scalar_type>(0) ? -b : b );
        else
            return ( ms < static_cast<scalar_type>(0) ? b : -b );
    }



    // change the weight of a new or existing sample, in such way that
    // the KKT conditions will be met. The action type can be either add or remove, 
    // this depends on whether a new or existing point's weight is modified. 
    // The direction is the direction of the change of weight.
    template<typename action_type, typename direction>
    inline
    void satisfy_KKT_conditions( std::size_t index, key_type key, double init_weight ) {


//      with an enum, or with a source= and destination= type of handling?
//      perhaps the latter, because it seems to be a more elegant solution     
//      enum { end_to_margin, end_to_error, end_to_remaining,
//             margin_to_remaining, margin_to_error, margin_to_error_star, 
//             remaining_to_margin, error_to_margin, error_star_to_margin } action_enum;


        vector_type margin_sense( base_type::key_lookup.size() );

        //vector_type maximum_values( margin_sense.size() );

        vector_type coef_sense;
        scalar_type weight_t = init_weight;

        // create a candidate column of the design matrix; this column also includes the
        // point under consideration
        vector_type candidate_column( base_type::key_lookup.size() );

	base_type::fill_kernel( key, base_type::key_lookup.begin(), base_type::key_lookup.end(),
                                candidate_column.begin() );

        if (debug)
            std::cout << "index: " << index << std::endl;

        int migrate_action = 99;
        while( migrate_action > 2 ) {


            if (debug)
                std::cout << "weight_t -------> " << weight_t;
            if (debug)
                std::cout << "    direction: " << direction::value << std::endl;

            // compute all coefficient sensitivities
            coef_sense.resize( margin_set.size()+1, false );
            ublas::matrix_range< ublas::matrix<double> > R_range( R.view() );
            ublas::symmetric_adaptor< ublas::matrix_range< ublas::matrix<double> > > R_view( R_range );
            atlas::symv( R_view, H.row(index), coef_sense );

// 	    for( int i=0; i<margin_set.size(); ++i ) {
// 		std::cout << "margin " << margin_set[i] << " coef sense " << coef_sense[i+1];
// 		std::cout << "  weight " << base_type::weight[i] << std::endl;
// 	    }

            // compute all margin sensitivities
            atlas::gemv( H.view(), coef_sense, margin_sense );
            atlas::xpy( candidate_column, margin_sense );

            scalar_type delta_weight_t;
            scalar_type max_var;
            int migrate_index;
            int migrate_index_2;


	    if ( margin_sense[index] < 0.0 ) {
	    	std::cout << "Something seems to be wrong, algorithm will get stuck now" << std::endl;
		std::cout << "Current index: " << index << std::endl;
		std::cout << "Matrix R: " << R.view() << std::endl;
// 		int qqq;
// 		std::cin >> qqq;
	    }

            //
            // Check for cases 1 and 2: end case for the incremental algorithm
            //
            if ( action_type::increment::value ) {
                // current point directly to the margin set?
                // -> INCREMENTAL ALGORITHM
                delta_weight_t = switch_add_sub<direction>(residual[index],epsilon) / margin_sense[index];

                if (debug)
                    std::cout << margin_sense[index] << std::endl;
                if (debug)
                    std::cout << residual[index] << std::endl;
                if (debug)
                    std::cout << switch_add_sub<direction>(residual[index],epsilon) << std::endl;


                migrate_action = 0;
                migrate_index = 0;
                migrate_index_2 = index;

                // current point directly to the error or error star set?
                // -> INCREMENTAL ALGORITHM
                if ( mpl::equal_to<direction,mpl::int_<1> >::type::value )
                    max_var = C - weight_t;
                else
                    max_var = -C - weight_t;

                if (switch_compare<direction>(max_var,delta_weight_t)) {
                    migrate_action = 1;
                    delta_weight_t = max_var;
                    migrate_index = 0;
                    migrate_index_2 = index;
                }
                // 		std::cout << "looked at the last point" << std::endl;

            }

            // -> DECREMENTAL ALGORITHM
            if ( action_type::decrement::value ) {
                max_var = -weight_t;
                if (switch_compare<direction>(max_var,delta_weight_t)) {
                    //if ( max_var < delta_weight_t ) {
                    migrate_action = 2;
                    delta_weight_t = max_var;
                    migrate_index = 0;
                    migrate_index_2 = index;
                }
            }

            // Check for migration from margin set to any of the other sets (remainder,error,error-star)
            for( unsigned int i=0; i < margin_set.size(); ++i ) {

                if ( switch_compare<direction>( 0.0, coef_sense[i+1] )) {
                    if (base_type::weight[i]<0.0) {
                        // Check for migration from margin set to remainder set
                        max_var = -base_type::weight[i] / coef_sense[i+1];
                        if (switch_compare<direction>(max_var,delta_weight_t)) {
                            migrate_action = 3;
                            delta_weight_t = max_var;
                            migrate_index = i;
                            migrate_index_2 = margin_set[i];
                        }
                    } else {
                        // Check for migration from margin set to error set
                        max_var = (C-base_type::weight[i]) / coef_sense[i+1];
                        if (switch_compare<direction>(max_var,delta_weight_t)) {
                            migrate_action = 4;
                            delta_weight_t = max_var;
                            migrate_index = i;
                            migrate_index_2 = margin_set[i];
                        }
                    }
                } else {
                    if (base_type::weight[i]>0.0) {
                        // Check for migration from margin set to remainder set
                        max_var = -base_type::weight[i] / coef_sense[i+1];
                        if (switch_compare<direction>(max_var,delta_weight_t)) {
                            migrate_action = 3;
                            delta_weight_t = max_var;
                            migrate_index = i;
                            migrate_index_2 = margin_set[i];
                        }
                    } else {
                        // Check for migration from margin set to error-star set
                        max_var = (-C-base_type::weight[i]) / coef_sense[i+1];
                        if (switch_compare<direction>(max_var,delta_weight_t)) {
                            migrate_action = 5;
                            delta_weight_t = max_var;
                            migrate_index = i;
                            migrate_index_2 = margin_set[i];
                        }
                    }
                }
            }

            // Check for migration from error set to margin set
            for( unsigned int i=0; i<error_set.size(); ++i ) {
                int idx = error_set[i];
                if ( switch_compare<direction>( 0.0, margin_sense[idx] )) {

                    max_var = (residual[idx] - epsilon) / margin_sense[idx];

                    if (debug)
                        if (max_var==0.0)
                            std::cout << "MAX VAR ==0 " << std::endl;

                    if ( switch_compare<direction>(max_var,delta_weight_t) ) {
                        migrate_action = 6;
                        delta_weight_t = max_var;
                        migrate_index = i;
                        migrate_index_2 = error_set[i];
                    }
                }
            }

            // Check for migration from error-star set to margin set
            for( unsigned int i=0; i<error_star_set.size(); ++i ) {
                int idx = error_star_set[i];
                if ( switch_compare<direction>( margin_sense[idx], 0.0 )) {
                    max_var = (residual[idx] + epsilon) / margin_sense[idx];
                    if (switch_compare<direction>(max_var,delta_weight_t)) {
                        migrate_action = 7;
                        delta_weight_t = max_var;
                        migrate_index = i;
                        migrate_index_2 = error_star_set[i];
                    }
                }
            }

            // Check for migration from the remainder set to the margin set
            for( unsigned int i=0; i<remaining_set.size(); ++i ) {
                int idx = remaining_set[i];
                max_var = (residual[idx] + inline_qm<direction>(margin_sense[idx],epsilon)) / margin_sense[idx];
                if (switch_compare<direction>(max_var,delta_weight_t)) {
                    migrate_action = 8;
                    delta_weight_t = max_var;
                    migrate_index = i;
                    migrate_index_2 = remaining_set[i];
                }
            }

            if (debug)
                std::cout << "delta weight_t:            " << delta_weight_t << std::endl;
            if (debug)
                std::cout << "Migration action chosen:   " << migrate_action << std::endl;
            if (debug)
                std::cout << "Affected index:            " << migrate_index << std::endl;
            if (debug)
                std::cout << "Affected index 2:          " << migrate_index_2 << std::endl;
            if (debug)
                std::cout << "Next candidate for weight: " << weight_t + delta_weight_t << std::endl;


            if (debug)
                if (delta_weight_t == 0.0 ) {
                    std::cout << "DELTA WEIGHT == 0! " << std::endl;
                    int qqq;
                    std::cin >> qqq;
                }

            weight_t += delta_weight_t;
            // std::cout << "update weight_t to " << weight_t << std::endl;

            // update weight vector and bias
            atlas::axpy( delta_weight_t, ublas::vector_range<vector_type>(coef_sense,ublas::range(1,coef_sense.size())), base_type::weight );
            base_type::bias += coef_sense(0) * delta_weight_t;

            // update residual vector
	    atlas::axpy( -delta_weight_t, margin_sense, residual );

          
	    
	    
	    // TODO give the cases names instead of these numbers
            switch( migrate_action ) {

                // Stopping criteria: case 0 and case 1
            case 0: {
                    // End-condition reached. The point under consideration is
                    // added to the margin set, and the algorithm will terminate.
                    // new sample -> margin set
                    add_to_margin( index );
		    base_type::weight.push_back( weight_t );
                    if (debug)
                        std::cout << "moved " << index << " to the margin set" << std::endl;
                    break;
                }

            case 1: {
                    // End-condition reached. The point under consideration is added
                    // to the error set or error star set, and the algorithm will terminate.
                    // new sample -> error/error-star set
                    if ( mpl::equal_to<direction,mpl::int_<1> >::type::value ) {
                        error_set.push_back( index );
                        if (debug)
                            std::cout << "moved " << index << " to the error set" << std::endl;
                    } else {
                        error_star_set.push_back( index );
                        if (debug)
                            std::cout << "moved " << index << " to the error-star set" << std::endl;
                    }
                    break;
                }

            case 2: {
                    // Decremental algorithm end condition reached. The point under consideration
                    // is removed from the margin set and added to the remaining set.
                    // At this point, it is alreay removed from the margin set.

                    std::cout << "not much to do... " << std::endl;

                    // we're at the end-point of the decremental algorithm!
                    // we could do all kinds of stuff: remove it from the all_vectors,
                    // from the residual and H matrix.

                    // yet to decide on

                    break;
                }

            case 3: {
                    // A sample has to migrate from the margin set to the
                    // remaining set (weight=0)
                    remove_from_margin( migrate_index );
                    remaining_set.push_back(migrate_index_2);
                    if (debug)
                        std::cout << "moved " << migrate_index_2 << " from the margin set to the remaining set" << std::endl;
                    break;
                }


            case 4: {
                    // A sample has to migrate from the margin set to the error set (weight=C)
                    remove_from_margin( migrate_index );
                    error_set.push_back(migrate_index_2);
                    if (debug)
                        std::cout << "moved " << migrate_index_2 << " from the margin set to the error set" << std::endl;
                    break;
                }

            case 5: {
                    // A sample has to migrate from the margin set to the error star set (weight=-C)
                    remove_from_margin( migrate_index );
                    error_star_set.push_back(migrate_index_2);
                    if (debug)
                        std::cout << "moved " << migrate_index_2 << " from the margin set to the error-star set" << std::endl;
                    break;
                }

            case 6: {
                    // A sample has to migrate from the error set to the margin set, while
                    // temporarily keeping its weight (=C)
                    // error set -> margin set
                    add_to_margin( migrate_index_2 );
		    // quick (constant time) removal: swap with latest element in vector
		    error_set[ migrate_index ] = error_set.back();
		    error_set.pop_back();
		    base_type::weight.push_back( C );
                    if (debug)
                        std::cout << "moved " << migrate_index_2 << " from the error set to the margin set" << std::endl;
                    break;
                }

            case 7: {
                    // A sample has to migrate from the error star set to the margin set, while
                    // temporarily keeping its weight (=-C)
                    // error star set -> margin set
                    add_to_margin( migrate_index_2 );
		    // quick (constant time) removal: swap with latest element in vector
		    error_star_set[ migrate_index ] = error_star_set.back();
		    error_star_set.pop_back();
		    base_type::weight.push_back( -C );
                    if (debug)
                        std::cout << "moved " << migrate_index_2 << " from the error-star set to the margin set" << std::endl;
                    break;
                }

            case 8: {
                    // A sample has to migrate from the remaining set to the margin set, while
                    // temporarily keeping its weight (=0)
                    // remaining set -> margin set
                    add_to_margin( migrate_index_2 );
		    // quick (constant time) removal: swap with latest element in vector
		    remaining_set[ migrate_index ] = remaining_set.back();
		    remaining_set.pop_back();
		    base_type::weight.push_back( 0.0 );
                    if (debug)
                        std::cout << "moved " << migrate_index_2 << " from the remaining set to the margin set" << std::endl;
                    break;
                }

            default: {
                    std::cout << "BUG IN AOSVR ROUTINE!!!" << std::endl;
                    int qqq;
                    std::cin >> qqq;
                    break;
                }
            }
        }


        if (debug)
            std::cout << std::endl;
    }




    // if a point is added to the margin set, additional actions have to be taken
    // the inverse matrix has to be updated
    // the system/design matrix has to be updated
    void add_to_margin( int index ) {

        std::size_t old_size = R.size1();
        std::size_t new_size = old_size + 1;
        key_type key = base_type::key_lookup[ index ];

        // adjust inverse matrix
        if (new_size==2) {

            // initialise the R matrix
            R.grow_row_column();
            R.matrix(0,0) = base_type::kernel( key, key );
            R.matrix(1,0) = -1.0;
            R.matrix(1,1) = 0.0;

        } else {
            // grow the R matrix
            R.grow_row_column();

            // fetch a view into the matrix _without_ the new row and columns
            ublas::matrix_range< ublas::matrix<double> > R_range( R.shrinked_view() );
            ublas::symmetric_adaptor< ublas::matrix_range< ublas::matrix<double> > > R_symm_view( R_range );

            // fetch a view into the last row of the matrix of the _old_ size
	    ublas::matrix_vector_slice< ublas::matrix<double> > R_row_part( R.shrinked_row(old_size) );

            // compute the unscaled last row of R (similar to the coefficient sensitivities)
            atlas::symv( R_symm_view, H.row(index), R_row_part );

            // compute the scaling factor

            // BAIL OUT HERE IF NECESSARY

            double divisor = base_type::kernel( key, key ) + atlas::dot( H.row(index), R_row_part );

	    if (std::fabs( divisor ) < 1e-12 ) {
	    	
		// from http://bach.ece.jhu.edu/pub/gert/slides/kernel.pdf
		// When the incremental inverse jacobian is (near) ill conditioned, a direct L1-norm minimisation
		// of the \alpha coefficients yields an optimally sparse solution
		
		std::cout << std::endl;
		std::cout << "k_tt + dot( H.row(index), R_row_part ) = " << divisor << std::endl;
	    	std::cout << "Problem detected, algorithm will get stuck real soon now." << std::endl;
		std::cout << "Point to be added to the margin set: " << index << std::endl;
		std::cout << H.row(index) << std::endl;
		std::cout << R_row_part << std::endl;
		
		for( unsigned int i=0; i<margin_set.size(); ++i ) {
			std::cout << margin_set[i] << " ";
		}
		std::cout << std::endl << std::endl;
		
		int qqq;
		std::cin >> qqq;
		
		//atlas::set( 0.0, R_row_part );
		//R.matrix( old_size, old_size ) = 0.0;
		
		
	    }

            // the divisor is within acceptable bounds; increase the R matrix as usual
            R.matrix(old_size,old_size) = -1.0 / divisor;

            // perform a rank-1 update of the R matrix
            atlas::syr( R.matrix(old_size,old_size), R_row_part, R_symm_view );

            // scale the last row with the scaling factor
            atlas::scal( R.matrix(old_size,old_size), R_row_part );
        }

        // adjust "design" matrix
        // NOTE has to be done AFTER the update of the R matrix!!  (to check ... )
        // because a row of the "old" design matrix is used in the determination of "delta", see above.
        H.grow_column();
	//ublas::matrix_column<ublas::matrix<double> >::iterator 
	base_type::fill_kernel( key, base_type::key_lookup.begin(), base_type::key_lookup.end(), 
                                ublas::column( H.matrix, old_size ).begin() );

//         for( unsigned int i=0; i<base_type::support_vector.size(); ++i )
//             H.matrix(i,old_size) = base_type::kernel( base_type::support_vector[i],base_type::support_vector[idx] );

        // perform the actual set transition
	margin_set.push_back( index );
	margin_key.push_back( key );
    }


    // input: the index in the margin key vector
    void remove_from_margin( unsigned int idx ) {

        // remove from design matrix
        H.swap_remove_column( idx+1 );

        // remove from the inverse matrix
        if (R.size1()==2) {
            R.remove_row_col(1);
            R.matrix(0,0)=0.0;
        } else {

	    ublas::matrix_range< ublas::matrix<double> > R_range( R.view() );
            ublas::symmetric_adaptor< ublas::matrix_range< ublas::matrix<double> > > R_view( R_range );

            vector_type R_row( row(R_view, idx+1) );
            atlas::syr( -1.0/R.matrix(idx+1,idx+1), R_row, R_view );

            // efficient removal from inverse matrix R
	    R.swap_remove_row_col( idx+1 );
        }
	
	// constant time removal from margin set index vector
        margin_set[ idx ] = margin_set.back();
        margin_set.pop_back();

	// constant time removal from margin set key vector
	margin_key[ idx ] = margin_key.back();
	margin_key.pop_back();

        // constant time removal from weight vector
	base_type::weight[ idx ] = base_type::weight.back();
	base_type::weight.pop_back();
    }


    // local memory

    static const bool debug = false;

    matrix_view< ublas::matrix<double> > H;					// (part of) design matrix H
    symmetric_view< ublas::matrix<double> > R;					// matrix inverse
    std::vector<double> residual;						// in some sense, the history of outputs

    scalar_type epsilon;
    scalar_type C;
    

    typedef typename base_type::index_type index_type;
    
    /*! A vector containing the indices into key_lookup of the margin vectors (-C < weight < C)*/
    std::vector< index_type > margin_set;

    /*! A vector containing the keys of the margin vectors (-C < weight < C)*/
    std::vector< key_type > margin_key;
    
    /*! A vector containing the indices into key_lookup of the error vectors (weight > C)*/
    std::vector< index_type > error_set;
    
    /*! A vector containing the indices into key_lookup of the error-star vectors (weight < -C)*/
    std::vector< index_type > error_star_set;
    
    /*! A vector containing the indices into key_lookup of the remaining vectors (weight == 0)*/
    std::vector< index_type > remaining_set;
};



/*!

On-line SVM classification algorithm. 

Duplicate data points (we're not yet discussing linear dependent points) cause the original algorithm to 
malfunction. This could be solved by introducing a weight per point, with the weight the number of 
times a point is present the data set.

Of course, the remaining points could be never measured at all: so we don't know (or care) whether duplicates 
are present in that kind of points. 

The original algorithm will crash only when a duplicate point enters the margin_set 

The algorithm will be able to detect whether a duplicate is being entered when it runs the 
satisfy_KKT_conditions; the distance between a new point and an existing point will be 0. 

In case of duplicate points, a weight of a prior point could be increased by 1 to represent that point. In 
case of linear dependent points; the weight of a number of prior points could be increased by the projection
weights (and that linear dependent point can be left out).

\todo
- introduce weighted examples
- be able to process contradicting examples (weight==0, then answer indeterminate or something)
- Unlearning of examples

\section bibliography2 References

-# Gert Cauwenberghs and Tomaso Poggio. Incremental and Decremental Support Vector
   Machine Learning. In Todd Leen and Thomas Dietterich and Volker Tresp, editors,
   Advances in Neural Information Processing Systems (NIPS'00).
   http://tinyurl.com/7cydn

*/


template< typename Problem, typename Kernel, typename PropertyMap >
class online_svm< Problem, Kernel, PropertyMap, typename boost::enable_if< is_classification<Problem> >::type >:
public kernel_machine< Problem, Kernel, PropertyMap > {
public:

    typedef kernel_machine< Problem, Kernel, PropertyMap > base_type;
    typedef typename base_type::kernel_type kernel_type;
    typedef typename base_type::result_type result_type;
    typedef typename Problem::input_type input_type;
    typedef typename Problem::output_type output_type;
    typedef typename Problem::example_type example_type;
    typedef double scalar_type;
    typedef ublas::symmetric_matrix<double> symmetric_type;
    typedef ublas::matrix<double> matrix_type;
    typedef ublas::vector<double> vector_type;

	typedef typename boost::property_traits<PropertyMap>::key_type key_type;
	typedef typename boost::property_traits<PropertyMap>::value_type object_type;


    online_svm( typename boost::call_traits<scalar_type>::param_type max_weight,
                typename boost::call_traits<kernel_type>::param_type k,
                typename boost::call_traits<PropertyMap>::param_type map ):
    base_type(k,map), C(max_weight) {}
        
    
    template< typename TokenIterator >
    online_svm( TokenIterator const begin, TokenIterator const end, 
                typename boost::call_traits<kernel_type>::param_type k,
                typename boost::call_traits<PropertyMap>::param_type map ):
		base_type(k,map) {
		C = 10.0;
		TokenIterator token( begin );
		if ( token != end ) {
			C = boost::lexical_cast<double>( *token++ );
		}
	}
    
    
    
    
    output_type operator()( typename boost::call_traits<input_type>::param_type x ) {
    	return (evaluate_f(x) >= 0.0);
    }
    
    // a non-classifying prediction function
    // is needed to compute condition figures
    scalar_type evaluate_f( typename boost::call_traits<input_type>::param_type x ) {
        scalar_type result(base_type::bias);
        // temp_K[i], not temp_K[i+1]
        if (margin_set.size()>0) {
            vector_type temp_K( margin_set.size() );
            base_type::fill_kernel( x, margin_key.begin(), margin_key.end(), temp_K.begin() );
            result += atlas::dot( temp_K, base_type::weight );
        }
        if (error_set.size()>0) {
            scalar_type temp_K(0);
            for( unsigned int i=0; i < error_set.size(); ++i ) {
		       key_type key = base_type::key_lookup[error_set[i]];
		        if ( (*base_type::data)[key].get<1>() )
                    temp_K += base_type::kernel( x, key );
                else
                    temp_K -= base_type::kernel( x, key );
           }
            result += C * temp_K;	   
	}
	return result;
    }
    
        

/*! learn the entire range of keys indicated by this range */
    template<typename KeyIterator>
    void learn( KeyIterator begin, KeyIterator end ) {
		KeyIterator key_iterator(begin);
		while( key_iterator != end ) {
			increment(*key_iterator);
			++key_iterator;
		}


	}


	float bool_to_float( bool const value ) {
		return ( value ? 1.0 : -1.0 );
	}



    /*! \param input an input pattern of input type I
        \param output an output pattern of output type O */
    void increment( key_type const key ) {

	   debug = false;
	
	std::size_t index = base_type::key_lookup.size();
	base_type::key_lookup.push_back( key );

        // record prediction error
        if (debug) {
            std::cout << "Starting AOSVR incremental algorithm for key " << key << std::endl;
	    	std::cout << "              the associated index is        " << index << std::endl;
		}
        

	//if (index > 581) debug=true;
	
	
	if (debug)
	std::cout << "Starting incremental SVM classification algorithm with " << index << " prior points" << std::endl;
	
		//if (index > 615 ) debug = true; else debug = false;
	
        //if (debug)

	// record condition figure
	if ( (*base_type::data)[key].get<1>() )
        	condition.push_back( evaluate_f((*base_type::data)[key].get<0>()) - 1.0 );
        else
        	condition.push_back( -evaluate_f((*base_type::data)[key].get<0>()) - 1.0 );
        
	// store the input and output 
	//base_type::support_vector.push_back( input );
	//outputs.push_back( output );
	
	
/*	std::cout << "Predicting history..." << std::endl;
	for( int i=0; i<base_type::support_vector.size(); ++i) 
		std::cout << i << ": " << evaluate_f( base_type::support_vector[i] ) << std::endl;*/
	
	
// 	if (debug)
// 	   std::cout << "The output is " << output << std::endl;

        // add to support vector buffer
        //     preserved_resize( all_vectors, index+1, x_t.size() );
        //     row( all_vectors, index ).assign( x_t );
        //     preserved_resize( h, index+1 );


        if ( index == 0 ) {

            // first point, initialise machine
            if (debug)
                std::cout << "Initialising the Accurate Online Support Vector Machine" << std::endl;

	    // set f(x_i) to y_i
            base_type::bias = bool_to_float( (*base_type::data)[key].get<1>() );
            condition.back() = 0.0;
            
	    // put this first point (with index 0) in the remaining set
	    remaining_set.push_back( 0 );

            // initialise the inverse matrix with a zero only
	    R.grow_row_column();
            R.matrix(0,0) = 0.0;

            // initialise "design" matrix with the value associated with the output
            H.grow_row_column();
    	    H.matrix(0,0) = bool_to_float( (*base_type::data)[key].get<1>() );

	    if (debug) 
	    	std::cout << std::endl;

        } else {


            // resize (view in) design matrix
            H.grow_row();

            if (debug) {
                std::cout << "H is now " << H.size1() << " by " << H.size2();
		std::cout << " #conditions " << condition.size();
		std::cout << " #weights " << base_type::weight.size();
		std::cout << " #keys    " << base_type::key_lookup.size() << std::endl;
	    }

            // fill last ROW of matrix H with this input sample
	    // TODO this should be an optimised function call
            //H.matrix( index, 0 ) = ( output ? 1.0 : -1.0 );

	    ublas::matrix_row<ublas::matrix<double> >::iterator j = ublas::row( H.matrix, index ).begin(); 
	    *j++ = bool_to_float( (*base_type::data)[key].get<1>() );
	    base_type::fill_kernel( key, margin_key.begin(), margin_key.end(), j );
	    
// 	    std::cout << "Filled H to ";
//     	    std::cout << H.view() << std::endl;

	    	    
	    if (debug)
	    	std::cout << "condition[index] is " << condition.back() << std::endl;
	    if ( condition.back() >= 0.0 ) {
		if (debug) 
			std::cout << "adding to remaining set..." << std::endl;	    
	        remaining_set.push_back( index );
                if (debug)
			std::cout << std::endl;
	    } else {
		if (debug)
			std::cout << "Satisfying KKT conditions... " << std::endl;
	    	satisfy_KKT_conditions( index, key, 0.0 );
	    }
	}

    
/*	std::cout << "Predicting history..." << std::endl;
	for( int i=0; i<base_type::support_vector.size(); ++i) 
		std::cout << i << ": " << evaluate_f( base_type::support_vector[i] ) << std::endl;*/
    
    	if (debug) {
	std::cout << "-------------------------------------------------------------------------------" << std::endl;
//  	int qqq;
//  	std::cin >> qqq;
	}
    
    }


    
    
    
    
    
    inline
    void satisfy_KKT_conditions( int index, key_type key, double init_weight ) {

        scalar_type weight_t = init_weight;
        	
	// create a candidate COLUMN of design matrix H; this column also includes the point under consideration
        
	//
	// TODO
	// TODO Duplicate detection can be done right here (distance function using kernels etc.)
	// TODO
	//

        vector_type candidate_column( base_type::key_lookup.size() );
	base_type::fill_kernel( key, base_type::key_lookup.begin(), base_type::key_lookup.end(),
                                candidate_column.begin() );

// 	if ( (*base_type::data)[key].get<1>() ) {
// 		for( unsigned int i=0; i<base_type::key_lookup.size(); ++i )
// 		  if ( (*base_type::data)[base_type::key_lookup[i]].get<1>() )
// 		  			 std::cout << base_type::kernel( key, base_type::key_lookup[i] ) << std::endl;
// 		  else 
// 		  			 std::cout << -base_type::kernel( key, base_type::key_lookup[i] ) << std::endl;
// 	} else {
// 		for( unsigned int i=0; i<base_type::key_lookup.size(); ++i )
// 		  if ( (*base_type::data)[base_type::key_lookup[i]].get<1>() )
// 		  			 std::cout << -base_type::kernel( key, base_type::key_lookup[i] ) << std::endl;
// 		  else 
// 		  			 std::cout << base_type::kernel( key, base_type::key_lookup[i] ) << std::endl;
// 	}
	if (debug)
		std::cout << "Computed candidate column" << std::endl;
// 	if (debug)
// 		std::cout << "Candidate column is " << candidate_column << std::endl;		
	
	// initialise the margin and coefficient sensitivity vectors
	vector_type margin_sense( base_type::key_lookup.size() );
        vector_type coef_sense;
        
        int migrate_action = 99;
        while( migrate_action > 2 ) {
            
	    
	    // Equation 10: compute all coefficient sensitivities
            coef_sense.resize( margin_set.size()+1, false );
            ublas::matrix_range< ublas::matrix<double> > R_range( R.view() );
            ublas::symmetric_adaptor< ublas::matrix_range< ublas::matrix<double> > > R_view( R_range );
            atlas::symv( R_view, H.row(index), coef_sense );
// 	    if (debug)
//       	        std::cout << "Coefficient sensitivities: " << coef_sense << std::endl;
// 	
            // Equation 12: compute all margin sensitivities
            atlas::gemv( H.view(), coef_sense, margin_sense );
            atlas::xpy( candidate_column, margin_sense );
// 	    if (debug)
// 	    	std::cout << "Margin sensitivities: " << margin_sense << std::endl;
		
// 	    if (debug) {
// 	        std::cout << "Condition numbers: ";
// 	 	for( unsigned int i=0;i<condition.size();++i) std::cout << condition[i] << ",";
// 	    	std::cout << std::endl;
// 	    }
		            
	    // seek for the largest possible increment in weight[index]
	    scalar_type delta_weight_t(0);
            scalar_type max_var;
            int migrate_index;
            int migrate_index_2;
	    	
            
            if (debug)
                std::cout << "weight_t -------> " << weight_t << std::endl;
            
	    // TODO
	    // BUG!! Use adult-1a.data to get this bug to work
	    // TODO
	    if ( margin_sense[index] < 0.0 ) {
	    	std::cout << "Something seems to be wrong, algorithm will get stuck now" << std::endl;
		std::cout << "Current index: " << index << std::endl;
		std::cout << "Matrix R: " << R.view() << std::endl;
		int qqq;
		std::cin >> qqq;
	    }
	    
            // CHECK END CONDITIONS OF CURRENT POINT
            // -> INCREMENTAL ALGORITHM
	    // current point directly to the margin set?
	    delta_weight_t = -condition[index] / margin_sense[index];
	    if (debug)
	    std::cout << "point -> margin: " << delta_weight_t << std::endl;
            migrate_action = 0;
            migrate_index = 0;
            migrate_index_2 = index;

      	    // current point directly to the error set?
            max_var = C - weight_t;
    	    if (debug)
    	    std::cout << "point -> error: " << max_var << std::endl;
      	    if (max_var < delta_weight_t) {
               	migrate_action = 1;
               	delta_weight_t = max_var;
               	migrate_index = 0;
               	migrate_index_2 = index;
      	    }



            // Check for migration from margin set to any of the other sets (remainder,error)
            for( unsigned int i=0; i < margin_set.size(); ++i ) {

		if ( coef_sense[i+1] < 0.0 ) {
	                
			// Check for migration from margin set to remaining set
                 	max_var = -base_type::weight[i] / coef_sense[i+1];
                	if (debug || max_var > 1e10)
			   std::cout << "margin -> remaining: " << max_var << std::endl;
                 	if (max_var < delta_weight_t) {
                       	migrate_action = 3;
                       	delta_weight_t = max_var;
                       	migrate_index = i;
                       	migrate_index_2 = margin_set[i];
                 	}
		 } else {
                        
			// Check for migration from margin set to error set
			max_var = (C-base_type::weight[i]) / coef_sense[i+1];
                	if (debug || max_var > 1e10)
			   std::cout << "margin -> error: " << max_var << std::endl;
                	if (max_var < delta_weight_t) {
                            	migrate_action = 4;
                            	delta_weight_t = max_var;
                            	migrate_index = i;
                            	migrate_index_2 = margin_set[i];
               		}
		}
            }
	    
	    // Check for migration from error set to margin set
            for( unsigned int i=0; i<error_set.size(); ++i ) {
                int idx = error_set[i];
		
		if ( margin_sense[idx] > 0.0 ) {
		
                max_var = -condition[idx] / margin_sense[idx];
                	if (debug || max_var > 1e10)
			   std::cout << "error -> margin: " << max_var << std::endl;
                if ( max_var < delta_weight_t) {
                    migrate_action = 6;
                    delta_weight_t = max_var;
                    migrate_index = i;
                    migrate_index_2 = error_set[i];
                }
		}
            }


            // Check for migration from the remaining set to the margin set
            for( unsigned int i=0; i<remaining_set.size(); ++i ) {
                int idx = remaining_set[i];
		if (margin_sense[idx] < 0.0 ) {
		
                max_var = -condition[idx] / margin_sense[idx];
                	if (debug || max_var > 1e10)
			   std::cout << "remaining -> margin: " << max_var << std::endl;
                if (max_var < delta_weight_t) {
                    migrate_action = 8;
                    delta_weight_t = max_var;
                    migrate_index = i;
                    migrate_index_2 = remaining_set[i];
                }
		}
            }

            if (debug)
                std::cout << "delta weight_t:            " << delta_weight_t << std::endl;
            if (debug)
                std::cout << "Migration action chosen:   " << migrate_action << std::endl;
            if (debug)
                std::cout << "Affected index:            " << migrate_index << std::endl;
            if (debug)
                std::cout << "Affected index 2:          " << migrate_index_2 << std::endl;
            if (debug)
                std::cout << "Next candidate for weight: " << weight_t + delta_weight_t << std::endl;
		
		
	    // Update the weight of the point under consideration
	    weight_t += delta_weight_t;
            
	    // Equation 8: update bias using its coefficient sensitivities and the change in the weight_t
            base_type::bias += delta_weight_t * coef_sense[0];
            
	    // Equation 9: update weight vector using the coefficient sensitivities and the change in the weight_t
            atlas::axpy( delta_weight_t, ublas::vector_range<vector_type>(coef_sense,ublas::range(1,coef_sense.size())), base_type::weight );
	    
	    if (debug) {
	    	std::cout << "Updated weights to: ";
	    	for( unsigned int i=0; i<base_type::weight.size(); ++i ) std::cout << base_type::weight[i] << " ";
	    	std::cout << std::endl;
	    	std::cout << "Bias is now: " << base_type::bias << std::endl;
	    }

            // Equation 11: update condition vector
	    atlas::axpy( delta_weight_t, margin_sense, condition );

// 	    if (debug) {
// 		    std::cout << "Updated conditions to: ";
// 		    for( unsigned int i=0; i<condition.size(); ++i ) std::cout << condition[i] << " ";
// 		    std::cout << std::endl;
// 	    }
	    
	    
	    
	    // TODO give the cases names instead of these numbers
            switch( migrate_action ) {


            case 0: {
		    // End-condition reached. The point under consideration (located at index) is
                    // added to the margin set, and the algorithm will terminate.
                    // new sample -> margin set
                    add_to_margin( index );
		    base_type::weight.push_back( weight_t );
                    if (debug)
                        std::cout << "moved " << index << " to the margin set" << std::endl;
                    break;
                }

            case 1: {
                    // End-condition reached. The point under consideration is added
                    // to the error set or error star set, and the algorithm will terminate.
                    // new sample -> error set
                    error_set.push_back( index );
                    if (debug)
                        std::cout << "moved " << index << " to the error set" << std::endl;
                    break;
                }
            case 3: {
                    // A sample has to migrate from the margin set to the
                    // remaining set (weight=0)
                    remove_from_margin( migrate_index );
                    remaining_set.push_back(migrate_index_2);
                    if (debug)
                        std::cout << "moved " << migrate_index_2 << " from the margin set to the remaining set" << std::endl;
                    break;
                }


            case 4: {
                    // A sample has to migrate from the margin set to the error set (weight=C)
                    remove_from_margin( migrate_index );
                    error_set.push_back(migrate_index_2);
                    if (debug)
                        std::cout << "moved " << migrate_index_2 << " from the margin set to the error set" << std::endl;
                    break;
                }
		
            case 6: {
                    // A sample has to migrate from the error set to the margin set, while
                    // temporarily keeping its weight (=C)
                    // error set -> margin set
                    add_to_margin( migrate_index_2 );
		    // quick (constant time) removal: swap with latest element in vector
		    error_set[ migrate_index ] = error_set.back();
		    error_set.pop_back();
		    base_type::weight.push_back( C );
                    if (debug) 
                        std::cout << "moved " << migrate_index_2 << " from the error set to the margin set" << std::endl;
			//std::cout << delta_weight_t << std::endl;
                    break;
                }

	    case 8: {
                    // A sample has to migrate from the remaining set to the margin set, while
                    // temporarily keeping its weight (=0)
                    // remaining set -> margin set
                    add_to_margin( migrate_index_2 );
		    // quick (constant time) removal: swap with latest element in vector
		    remaining_set[ migrate_index ] = remaining_set.back();
		    remaining_set.pop_back();
		    base_type::weight.push_back( 0.0 );
                    if (debug)
                        std::cout << "moved " << migrate_index_2 << " from the remaining set to the margin set" << std::endl;
                    break;
                }


            default: {
                    std::cout << "BUG IN Incremental SVM ROUTINE!!!" << std::endl;
                    int qqq;
                    std::cin >> qqq;
                    break;
                }

	    }

//             if (debug) {
// 	    	int qqq;
// 		std::cin >> qqq;
// 	    	
// 	    }


  	    //migrate_action = 0;
	}
	

	if (debug)
		std::cout << std::endl;

    }
    

    // if a point is added to the margin set, additional actions have to be taken
    // the inverse matrix has to be updated
    // the system/design matrix has to be updated
    void add_to_margin( int idx ) {

        unsigned int old_size = R.size1();
        unsigned int new_size = old_size + 1;
        //std::cout << "associated key is " << base_type::key_lookup[idx] << std::endl;

	// if a point moves to the margin set, by definition, its condition equals 0.
	condition[ idx ] = 0.0;

        // adjust inverse matrix
        if (new_size==2) {

            // initialise the R matrix
            R.grow_row_column();
	    
	    // this is correct: Q_ii == K(x_i,x_i)
            R.matrix(0,0) = base_type::kernel( base_type::key_lookup[idx], base_type::key_lookup[idx] );
			// NEGATED bool_to_float (!!)
            R.matrix(1,0) = -bool_to_float( (*base_type::data)[base_type::key_lookup[idx]].get<1>() );
            R.matrix(1,1) = 0.0;

        } else {
            // grow the R matrix
            R.grow_row_column();

            // fetch a view into the matrix _without_ the new row and columns
            ublas::matrix_range< ublas::matrix<double> > R_range( R.shrinked_view() );
            ublas::symmetric_adaptor< ublas::matrix_range< ublas::matrix<double> > > R_symm_view( R_range );

            // fetch a view into the last row of the matrix of the _old_ size
            //ublas::matrix_row< ublas::matrix_range< ublas::matrix<double> > > R_row_part( R.shrinked_row(old_size) );
            ublas::matrix_vector_slice< ublas::matrix<double> > R_row_part( R.shrinked_row(old_size) );
	    
            // compute the unscaled last row of R (similar to the coefficient sensitivities)
            atlas::symv( R_symm_view, H.row(idx), R_row_part );

            // compute the scaling factor

            // BAIL OUT HERE IF NECESSARY
	    
	    double divisor = base_type::kernel( base_type::key_lookup[idx], base_type::key_lookup[idx] ) +
                                                  atlas::dot( H.row(idx), R_row_part );
	    
	    // Pseudoinverse of a SVD:
	    // in this case: fill that part of R with zeros! 
	    // see http://kwon3d.com/theory/jkinem/svd.html
	    // for values on the diagonal
	    // if w_i > epsilon, then w_i  = 1/w_i
	    // else w_i = 0
	    // i.e. infinity is replaced with 0.
	    // So, that row (and column) of R should become 0.
	    
	    // --->>> Sample 65 is identical to 582!!
	    // So, their distance is 0.
	    // So, the max_weight of 65 = 2C?!!
	    
	    // At the moment, let the algorithm crash, because adding 0 to R is not a final solution (although it works!).
	    if (std::fabs( divisor ) < 1e-12 ) {
	    	
		// from http://bach.ece.jhu.edu/pub/gert/slides/kernel.pdf
		// When the incremental inverse jacobian is (near) ill conditioned, a direct L1-norm minimisation
		// of the \alpha coefficients yields an optimally sparse solution
		
		std::cout << std::endl;
		std::cout << "k_tt + dot( H.row(idx), R_row_part ) = " << divisor << std::endl;
	    	std::cout << "Problem detected, algorithm will get stuck real soon now." << std::endl;
		std::cout << "Point to be added to the margin set: " << idx << std::endl;
		std::cout << H.row(idx) << std::endl;
		std::cout << R_row_part << std::endl;
		
		for( unsigned int i=0; i<margin_set.size(); ++i ) {
			std::cout << margin_set[i] << " ";
		}
		std::cout << std::endl << std::endl;
		
		int qqq;
		std::cin >> qqq;
		
		//atlas::set( 0.0, R_row_part );
		//R.matrix( old_size, old_size ) = 0.0;
		
		
	    }

            // the divisor is within acceptable bounds; increase the R matrix as usual
            R.matrix(old_size,old_size) = -1.0 / divisor;
						  
            // perform a rank-1 update of the R matrix
            atlas::syr( R.matrix(old_size,old_size), R_row_part, R_symm_view );

            // scale the last row with the scaling factor
            atlas::scal( R.matrix(old_size,old_size), R_row_part );
        
	}

	
// 	std::cout << "R is now: "<< R.view() << std::endl;
	
	
        // adjust "design" matrix
        // NOTE has to be done AFTER the update of the R matrix!!  (to check ... )
        // because a row of the "old" design matrix is used in the determination of "delta", see above.
        H.grow_column();

	base_type::fill_kernel( base_type::key_lookup[idx], base_type::key_lookup.begin(), base_type::key_lookup.end(),
                                ublas::column( H.matrix, old_size ).begin() );
                                
//     std::cout << H.view() << std::endl;

// 	if ( (*base_type::data)[base_type::key_lookup[idx]].get<1>() ) {
// 		for( unsigned int i=0; i<base_type::key_lookup.size(); ++i )
// 		  if ( (*base_type::data)[base_type::key_lookup[i]].get<1>() )
// 		  			 std::cout << base_type::kernel( base_type::key_lookup[idx], base_type::key_lookup[i] ) << std::endl;
// 		  else 
// 		  			 std::cout << -base_type::kernel( base_type::key_lookup[idx], base_type::key_lookup[i] ) << std::endl;
// 	} else {
// 		for( unsigned int i=0; i<base_type::key_lookup.size(); ++i )
// 		  if ( (*base_type::data)[base_type::key_lookup[i]].get<1>() )
// 		  			 std::cout << -base_type::kernel( base_type::key_lookup[idx], base_type::key_lookup[i] ) << std::endl;
// 		  else 
// 		  			 std::cout << base_type::kernel( base_type::key_lookup[idx], base_type::key_lookup[i] ) << std::endl;
// 	}
    
    
    
    
/*	if ( outputs[idx] ) {
		for( unsigned int i=0; i<base_type::support_vector.size(); ++i )
		  if ( outputs[i] )
            	     H.matrix(i,old_size) = base_type::kernel( base_type::support_vector[idx], base_type::support_vector[i] );
		  else 
            	     H.matrix(i,old_size) = -base_type::kernel( base_type::support_vector[idx], base_type::support_vector[i] );
	} else {
		for( unsigned int i=0; i<base_type::support_vector.size(); ++i )
		  if ( outputs[i] )
            	     H.matrix(i,old_size) = -base_type::kernel( base_type::support_vector[idx], base_type::support_vector[i] );
		  else 
            	     H.matrix(i,old_size) = base_type::kernel( base_type::support_vector[idx], base_type::support_vector[i] );
	}*/
	
// 	for( unsigned int i=0; i<base_type::support_vector.size(); ++i )
//             H.matrix(i,old_size) = base_type::kernel( base_type::support_vector[i],base_type::support_vector[idx] );

        // perform the actual set transition
	    margin_set.push_back( idx );
	    
	    margin_key.push_back( base_type::key_lookup[idx] );
	    
	    
    }



    void remove_from_margin( unsigned int idx ) {

        // remove from design matrix
        H.swap_remove_column( idx+1 );

        // remove from the inverse matrix
        if (R.size1()==2) {
            R.remove_row_col(1);
            R.matrix(0,0)=0.0;
        } else {

	    
	    double divisor = R.matrix(idx+1,idx+1);
	    
	    if ( std::fabs( divisor ) < 1e-12 ) {
	    	std::cout << "A problem occured when removing from the margin, small divisor detected." << std::endl;
	        std::cout << "Divisor value: " << divisor << std::endl;
		int qqq;
		std::cin >> qqq;
	    }
	    
	    
	    ublas::matrix_range< ublas::matrix<double> > R_range( R.view() );
            ublas::symmetric_adaptor< ublas::matrix_range< ublas::matrix<double> > > R_view( R_range );

            vector_type R_row( row(R_view, idx+1) );
            atlas::syr( -1.0/divisor, R_row, R_view );

            // efficient removal from inverse matrix R
	    R.swap_remove_row_col( idx+1 );
        }
	
	// constant time removal from margin set index vector
        margin_set[ idx ] = margin_set.back();
        margin_set.pop_back();
        
	// constant time removal from margin set key vector
	margin_key[ idx ] = margin_key.back();
	margin_key.pop_back();
        
        // constant time removal from weight vector
	base_type::weight[ idx ] = base_type::weight.back();
	base_type::weight.pop_back();
	
    }
    

    //
    // leave-one-out crossvalidation error estimate
    // section 3.1 of "Incremental and Decremental Support Vector Machine Learning"
    // 
    double loocv() {
    
	    // all remaining samples: "correct"
	    
	    for( unsigned int i=0; i<margin_set.size(); ++i ) {
	    	unsigned int idx = margin_set[i];
	    
	    	if (condition[idx] < -1.0) {
			// sample is "incorrect"
		} else {
		        // condition is >= -1.0:
			// perform decremental algorithm until a stopping criterion
			// satisfy_KKT_conditions( idx, base_type::weight[i] );
			
			if ( condition[idx] < -1.0 ) {
			   // sample is "incorrect"
			} else {
			   // sample is correct, because the other stopping criterion has been met
			   // i.e. the condition that weight[i]==0
			}
		}
	    }

	    	for( unsigned int i=0; i<error_set.size(); ++i ) {
	    		// do the same for the error vectors as done for the margin vectors
		}
	    
	    return 0.0;
	}

    
    //static const bool debug = false;
    bool debug;

    matrix_view< ublas::matrix<double> > H;					// "design matrix" Q (not as in the paper!)
    symmetric_view< ublas::matrix<double> > R;					// matrix inverse

    /*! vector "g" containing condition numbers for associated (indexed) vectors */
    std::vector<scalar_type> condition;						

    /*! maximum weight */
    scalar_type C;

    /*! A vector containing the indices of the margin vectors (y_i f(x_i) == 1) */
    std::vector<std::size_t> margin_set;

    /*! A vector containing the keys of the margin vectors ((y_i f(x_i) == 1) */
    std::vector<key_type> margin_key;
    
    /*! A vector containing the indices of the error vectors (exceeding the margin) */
    std::vector<std::size_t> error_set;
    
    /*! A vector containing the indices of the remaining vectors (within the margin) */
    std::vector<std::size_t> remaining_set;
    
};




// ON-LINE SVM RANKING ALGORITHM

// template<typename PropertyMap, typename Problem, typename Kernel>
// class online_svm<PropertyMap,Problem,Kernel,typename boost::enable_if< is_ranking<Problem> >::type >: 
//          public kernel_machine<PropertyMap,Problem,Kernel> {
// 
//     typedef kernel_machine<PropertyMap,Problem,Kernel> base_type;
//     typedef typename base_type::kernel_type kernel_type;
//     typedef typename base_type::result_type result_type;
//     typedef typename Problem::input_type input_type;
//     typedef typename Problem::output_type output_type;
//     typedef double scalar_type;
//     typedef ublas::symmetric_matrix<double> symmetric_type;
//     typedef ublas::matrix<double> matrix_type;
//     typedef ublas::vector<double> vector_type;
// 
//   typedef typename boost::range_value<input_type>::type point_type;
//   typedef typename output_type::const_iterator const_output_iterator;
//   typedef typename std::vector<input_type>::const_iterator const_svec_iterator;
// 
//   online_svm(typename boost::call_traits<kernel_type>::param_type k,
// 	     typename boost::call_traits<scalar_type>::param_type max_weight):
//   base_type(k), C(max_weight), inner_machine(k, max_weight) { }
// 
//   /*! \param input an input pattern of input type I
//       \param output an output pattern of output type O */
// 
//   
// 
//   void push_back(input_type const &input, output_type const &output) {
//     /* Here's how it works as you push a new vector in:
// 
//        1. Compare the vector, i, and its output to each <vector, output> pair we've already 
//           stored. This set could, of course, be empty.
//        2. For each vector j in our stored set which has an output not identical to ours, 
//           create a difference vector and add that difference vector to a new input set.
//        3. Call push_back on our internal classification SVM for each difference vector. */
//     
//     const_output_iterator y = ys.begin();
// 
//     for (const_svec_iterator i = base_type::support_vector.begin(); 
// 	 *y != ys.end() && i != base_type::support_vector.end(); 
// 	  ++y, ++i) {
//       if (*y != output) {
// 	input_type diff_vec;
// 	std::transform(input.begin(), input.end(), i->begin(), 
// 		       diff_vec.begin(), std::minus<point_type>()); 
// 	inner_machine.push_back(diff_vec, (output > *y));
//       }
//     }
//     base_type::support_vector.push_back(input);
//     ys.push_back(output);
//   }
//   std::vector<output_type> ys;
//   scalar_type C;
// 
//   typedef classification<input_type,bool> inner_problem;
//   online_svm<PropertyMap,inner_problem,Kernel> inner_machine;
// };



} //namespace kml

#endif

