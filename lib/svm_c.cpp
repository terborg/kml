/*****************************************************************************
 *  The Kernel-Machine Library                                               *
 *  Copyright (C) 2004, 2005 by Rutger W. ter Borg and Meredith L. Patterson *
 *                                                                           *
 *  This program is free software; you can redistribute it and/or            *
 *  modify it under the terms of the GNU General Public License              *
 *  as published by the Free Software Foundation; either version 2           *
 *  of the License, or (at your option) any later version.                   *
 *                                                                           *
 *  This program is distributed in the hope that it will be useful,          *
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of           *
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the            *
 *  GNU General Public License for more details.                             *
 *                                                                           *
 *  You should have received a copy of the GNU General Public License        *
 *  along with this program; if not, write to the Free Software              *
 *  Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307    *
 *****************************************************************************/

#ifndef _SVM_H
#define _SVM_H

#include <kml/svm.hpp>
#include <kml/gaussian.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/bindings/traits/std_vector.hpp>
#include <boost/numeric/bindings/traits/ublas_vector.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/mpl/equal_to.hpp>
#include <boost/type_traits/is_float.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/random_number_generator.hpp>
#include <boost/range/value_type.hpp>
#include <vector>

#include "kml/svm_c.h"

namespace ublas = boost::numeric::ublas;
namespace mpl = boost::mpl;

typedef kml::classification<std::vector<double>, int> class_double;
typedef kml::ranking<std::vector<double>, double> rank_double;
  
extern "C" {  
  void printweights(void *v) {
    kml::svm<class_double, kml::gaussian>* m = (kml::svm<class_double, kml::gaussian> *) v;
    m->printweights();
  }

  void* kml_new_classification_double_gaussian(double k, double s) { 
    return (void *) new kml::svm<class_double, kml::gaussian>(k, s);
  }

  void* kml_copy_classification_double_gaussian(void *v) {
    kml::svm<class_double, kml::gaussian>* m = (kml::svm<class_double, kml::gaussian> *) v;
    return (void *) new kml::svm<class_double, kml::gaussian>(*m);
  }

  void kml_delete_classification_double_gaussian(void *v) {
    delete (kml::svm<class_double, kml::gaussian>*) v;
  }

  void kml_learn_classification_double_gaussian(void *v, double **p, int *t,
						int sz_row, int sz_col) {
    kml::svm<class_double, kml::gaussian>* m = (kml::svm<class_double, kml::gaussian> *) v;
    std::vector<std::vector<double> > points;
    for (int j = 0; j < sz_row; ++j) {
      points.push_back(std::vector<double>(*p, (*p) + sz_col));
      ++p;
    }
    std::vector<int> target(t, t+sz_row);
    m->learn(points, target);
    std::cerr << "Back in the wrapper, done learning" << std::endl;
  }

  double kml_classify_double_gaussian(void *v, double *i, int sz) {
    kml::svm<class_double, kml::gaussian>* m = (kml::svm<class_double, kml::gaussian> *) v;
    if (m->operator()(std::vector<double>(i, i+sz)) > 0)
      return 1;
    else
      return 0;
  }

  void* kml_new_ranking_double_gaussian(double k, double s) {
    return (void *) new kml::svm<rank_double, kml::gaussian>(k, s);
  }

  void kml_delete_ranking_double_gaussian(void *v) {
    delete (kml::svm<rank_double, kml::gaussian>*) v;
  }

  void kml_learn_ranking_double_gaussian(void *v, double **p, int *t, int sz_row,
					 int sz_col) {

  }
  /*
  double kml_rank_double_gaussian(void* v, double *i, int sz) {

  }
  */
}

#endif
