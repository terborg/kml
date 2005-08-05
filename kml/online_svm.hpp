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

#include <kml/online_determinate.hpp>
#include <kml/matrix_view.hpp>
#include <kml/symmetric_view.hpp>



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

-# Gert Cauwenberghs and Tomaso Poggio. Incremental and Decremental Support Vector
   Machine Learning. In Todd Leen and Thomas Dietterich and Volker Tresp, editors,
   Advances in Neural Information Processing Systems (NIPS'00).
   http://tinyurl.com/7cydn
-# Junshui Ma et al., 2003. Accurate On-Line Support Vector Regression.
   \e Neural \e Computation, pp. 2700-2701.
   http://tinyurl.com/5g2bb
-# Mario Martin, 2002. On-line Support Vector Machine for Function Approximation.
   Technical report LSI-02-11-R, Universitat Politecnica de Catalunya
   http://tinyurl.com/48y9n
   
*/


template< typename I, typename O, template<typename,int> class K, class Enable = void >
class online_svm: public online_determinate<I,O,K> {};


//
//
// REGRESSION ALGORITHM
//
//

template< typename I, typename O, template<typename,int> class K>
class online_svm<I,O,K, typename boost::enable_if< boost::is_float<O> >::type >: 
         public online_determinate<I,O,K> {
public:
    typedef online_determinate<I,O,K> base_type;
    typedef typename base_type::kernel_type kernel_type;
    typedef typename base_type::result_type result_type;
    typedef I input_type;
    typedef O output_type;
    typedef double scalar_type;
    typedef ublas::symmetric_matrix<double> symmetric_type;
    typedef ublas::matrix<double> matrix_type;
    typedef ublas::vector<double> vector_type;


    // Bug fixed: use call_traits instead of const&, so can be called with both reference and non-reference
    /*! \param k the kernel construction parameter
        \param tube_width the error-insensitive tube width, parameter \f$ \epsilon \f$ of Vapnik's e-insensitive loss function
	\param max_weight the maximum weight assigned to any support vector, support vector machine's parameter C
    */
    online_svm( typename boost::call_traits<kernel_type>::param_type k,
                typename boost::call_traits<scalar_type>::param_type tube_width,
                typename boost::call_traits<scalar_type>::param_type max_weight ):
    base_type(k), epsilon(tube_width), C(max_weight) {}

    result_type operator()( input_type const &x ) {
        result_type result = base_type::bias;
        // temp_K[i], not temp_K[i+1]
        if (margin_set.size()>0) {
            vector_type temp_K( margin_set.size() );
            for( unsigned int i=0; i < margin_set.size(); ++i )
                temp_K[i] = base_type::kernel( base_type::support_vector[margin_set[i]], x );
            result += atlas::dot( temp_K, base_type::weight );
        }
        if (error_set.size()>0) {
            result_type temp_K(0);
            //vector_type temp_K( error_set.size() );
            for( unsigned int i=0; i < error_set.size(); ++i )
                temp_K += kernel( base_type::support_vector[error_set[i]], x );
            result += C * temp_K;
        }
        if (error_star_set.size()>0) {
            result_type temp_K(0);
            for( unsigned int i=0; i < error_star_set.size(); ++i )
                temp_K += kernel( base_type::support_vector[error_star_set[i]], x );
            result -= C * temp_K;
        }
        return result;
    }

    /*! \param input an input pattern of input type I
        \param output an output pattern of output type O */
    void push_back( input_type const &input, output_type const &output ) {


        int index = base_type::support_vector.size();

        // record prediction error
        if (debug)
            std::cout << "Starting AOSVR incremental algorithm with " << index << " prior points" << std::endl;

        // important: the sign of the residual is used below
        residual.push_back( output - operator()( input ) );
        base_type::support_vector.push_back( input );

        // add to support vector buffer
        //     preserved_resize( all_vectors, index+1, x_t.size() );
        //     row( all_vectors, index ).assign( x_t );
        //     preserved_resize( h, index+1 );


        if ( index == 0 ) {

            // first point, initialise machine
            if (debug)
                std::cout << "Initialising the Accurate Online Support Vector Regression machine" << std::endl;
            residual.back() = 0.0;
            base_type::bias = output;
            
	    // put this first point (with index 0) in the remaining set
	    remaining_set.push_back( 0 );

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
            H.matrix( index, 0 ) = 1.0;
            for( unsigned int i=0; i < margin_set.size(); ++i )
                H.matrix(index,i+1) = base_type::kernel( base_type::support_vector[margin_set[i]], input );


            if ( std::fabs(residual.back()) <= epsilon ) {
                if (debug)
                    std::cout << "adding to remaining set..." << std::endl;
                remaining_set.push_back( index );

            } else {
                if (debug)
                    std::cout << "re-establishing KKT conditions..." << std::endl;

                if ( residual[ index ] < 0.0 )
                    satisfy_KKT_conditions< incremental_type, mpl::int_<-1> >( index, 0.0 );
                else
                    satisfy_KKT_conditions< incremental_type, mpl::int_<1> >( index, 0.0 );
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
    void satisfy_KKT_conditions( int index, double init_weight ) {


//      with an enum, or with a source= and destination= type of handling?
//      perhaps the latter, because it seems to be a more elegant solution     
//      enum { end_to_margin, end_to_error, end_to_remaining,
//             margin_to_remaining, margin_to_error, margin_to_error_star, 
//             remaining_to_margin, error_to_margin, error_star_to_margin } action_enum;


        vector_type margin_sense( base_type::support_vector.size() );

        //vector_type maximum_values( margin_sense.size() );

        vector_type coef_sense;
        scalar_type weight_t = init_weight;

        // create a candidate column of the design matrix; this column also includes the
        // point under consideration
        vector_type candidate_column( base_type::support_vector.size() );
        for( unsigned int i=0; i<base_type::support_vector.size(); ++i )
            candidate_column[i] = base_type::kernel( base_type::support_vector[index], base_type::support_vector[i] );

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


	    // A swap on R, the matrix inverse view, must be possible ....  ! !! ! ! ! !



            // compute all margin sensitivities
            atlas::gemv( H.view(), coef_sense, margin_sense );
            atlas::xpy( candidate_column, margin_sense );

            scalar_type delta_weight_t;
            scalar_type max_var;
            int migrate_index;
            int migrate_index_2;




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
    void add_to_margin( int idx ) {

        unsigned int old_size = R.size1();
        unsigned int new_size = old_size + 1;


        // adjust inverse matrix
        if (new_size==2) {
            // initialise the R matrix
            R.grow_row_column();
            R.matrix(0,0) = base_type::kernel( base_type::support_vector[idx], base_type::support_vector[idx] );
            R.matrix(1,0) = -1.0;
            R.matrix(1,1) = 0.0;

        } else {
            // grow the R matrix
            R.grow_row_column();

            // fetch a view into the matrix _without_ the new row and columns
            ublas::matrix_range< ublas::matrix<double> > R_range( R.shrinked_view() );
            ublas::symmetric_adaptor< ublas::matrix_range< ublas::matrix<double> > > R_symm_view( R_range );

            // fetch a view into the last row of the matrix of the _old_ size
            ublas::matrix_row< ublas::matrix_range< ublas::matrix<double> > > R_row_part( R.shrinked_row(old_size) );

            // compute the unscaled last row of R (similar to the coefficient sensitivities)
            atlas::symv( R_symm_view, H.row(idx), R_row_part );

            // compute the scaling factor

            // BAIL OUT HERE IF NECESSARY

            R.matrix(old_size,old_size) = -1.0 / (base_type::kernel( base_type::support_vector[idx], base_type::support_vector[idx] ) +
                                                  atlas::dot( H.row(idx), R_row_part ));

            // perform a rank-1 update of the R matrix
            atlas::syr( R.matrix(old_size,old_size), R_row_part, R_symm_view );

            // scale the last row with the scaling factor
            atlas::scal( R.matrix(old_size,old_size), R_row_part );
        }

        // adjust "design" matrix
        // NOTE has to be done AFTER the update of the R matrix!!  (to check ... )
        // because a row of the "old" design matrix is used in the determination of "delta", see above.
        H.grow_column();
        for( unsigned int i=0; i<base_type::support_vector.size(); ++i )
            H.matrix(i,old_size) = base_type::kernel( base_type::support_vector[i],base_type::support_vector[idx] );

        // perform the actual set transition
	margin_set.push_back( idx );
    }



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

    /*! A vector containing the indices of the margin vectors (-C < weight < C)*/
    std::vector<unsigned int> margin_set;
    
    /*! A vector containing the indices of the error vectors (weight > C)*/
    std::vector<unsigned int> error_set;
    
    /*! A vector containing the indices of the error-star vectors (weight < -C)*/
    std::vector<unsigned int> error_star_set;
    
    /*! A vector containing the indices of the remaining vectors (weight == 0)*/
    std::vector<unsigned int> remaining_set;
};


// ON-LINE SVM RANKING ALGORITHM

template<typename I, typename O, template<typename,int> class K>
/* this will eventually take a problem_type, right? */
class online_svm<I,O,K, typename boost::enable_if< boost::is_same<O,int> >::type >:
    public online_determinate<I,O,K> {

  typedef online_determinate<I,O,K> base_type;
  typedef I input_type;
  typedef O output_type;
  typedef typename base_type::kernel_type kernel_type;
  typedef double scalar_type;
  typedef typename boost::range_value<input_type>::type point_type;
  typedef typename output_type::const_iterator const_output_iterator;
  typedef typename std::vector<input_type>::const_iterator const_svec_iterator;

  online_svm(typename boost::call_traits<kernel_type>::param_type k,
	     typename boost::call_traits<scalar_type>::param_type max_weight):
  base_type(k), C(max_weight), inner_machine(k, max_weight) { }

  /*! \param input an input pattern of input type I
      \param output an output pattern of output type O */

  

  void push_back(input_type const &input, output_type const &output) {
    /* Here's how it works as you push a new vector in:

       1. Compare the vector, i, and its output to each <vector, output> pair we've already 
          stored. This set could, of course, be empty.
       2. For each vector j in our stored set which has an output not identical to ours, 
          create a difference vector and add that difference vector to a new input set.
       3. Call push_back on our internal classification SVM for each difference vector. */
    
    const_output_iterator y = ys.begin();

    for (const_svec_iterator i = base_type::support_vector.begin(); 
	 *y != ys.end() && i != base_type::support_vector.end(); 
	  ++y, ++i) {
      if (*y != output) {
	input_type diff_vec;
	std::transform(input.begin(), input.end(), i->begin(), 
		       diff_vec.begin(), std::minus<point_type>()); 
	inner_machine.push_back(diff_vec, (output > *y));
      }
    }
    base_type::support_vector.push_back(input);
    ys.push_back(output);
  }
  std::vector<output_type> ys;
  scalar_type C;
  online_svm<I,bool,K> inner_machine;
};

//
//
// ON-LINE SVM CLASSIFICATION ALGORITHM
//
//

// it will become something like this:
// template<typename P, template<typename,int> class K>

template<typename I, typename O, template<typename,int> class K>
class online_svm<I,O,K, typename boost::enable_if< boost::is_same<O,bool> >::type >: 
         public online_determinate<I,O,K> {
	 
    typedef online_determinate<I,O,K> base_type;
    typedef I input_type;
    typedef O output_type;
    typedef typename base_type::kernel_type kernel_type;
    typedef double scalar_type;

    online_svm( typename boost::call_traits<kernel_type>::param_type k,
                typename boost::call_traits<scalar_type>::param_type max_weight ):
    base_type(k), C(max_weight) {}
    
    /*! \param input an input pattern of input type I
        \param output an output pattern of output type O */
    void push_back( input_type const &input, output_type const &output ) {

    	// if the output type is double, no conversion of +1, -1 to +1, -1 is needed?
	// if the output type is bool, some conversions are needed to +1, -1
    
	// This seems to be a MUCH simpler case than regression
	
	
        //vector_type margin_sense( base_type::support_vector.size() );

        //vector_type maximum_values( margin_sense.size() );

        //vector_type coef_sense;

	
	
	
	
	
	
	
	
	    
    }









    
    
    
    
    
    
    
    
    
    
    scalar_type C;

    /*! A vector containing the indices of the margin vectors (y_i f(x_i) == 1 */
    std::vector<unsigned int> margin_set;
    
    /*! A vector containing the indices of the error vectors (exceeding the margin) */
    std::vector<unsigned int> error_set;
    
    /*! A vector containing the indices of the remaining vectors (within the margin) */
    std::vector<unsigned int> remaining_set;
};







} //namespace kml

#endif




