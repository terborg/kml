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
#include <kml/linear.hpp>
#include <kml/polynomial.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/vector_property_map.hpp>
#include <vector>
#include <utility>
#include <iostream>

#include "kml/svm_c.h"

typedef std::vector<double> input_type;

typedef boost::tuple<input_type, double> class_type;
typedef boost::tuple<input_type, int, double> rank_type;

typedef boost::vector_property_map<class_type> class_property_map;
typedef boost::vector_property_map<rank_type> rank_property_map;

typedef kml::classification<class_type> class_prob;
typedef kml::ranking<rank_type> rank_prob;

typedef kml::gaussian<input_type> gaussian_k;
typedef kml::polynomial<input_type> polynomial_k;
typedef kml::linear<input_type> linear_k;

extern "C" {  

  void* kml_new_class_double_gaussian(double k, double s) { 
    boost::shared_ptr<class_property_map> m(new class_property_map);
    kml::svm<class_prob, gaussian_k, class_property_map> *v = new kml::svm<class_prob, gaussian_k, class_property_map>(k, s, m);
    return (void *) v;
    //    return (void *) new kml::svm<class_prob, gaussian_k, class_property_map>(k, s, m);
  }

  void* kml_copy_class_double_gaussian(void *v) {
    kml::svm<class_prob, gaussian_k, class_property_map>* m = (kml::svm<class_prob, gaussian_k, class_property_map> *) v;
    kml::svm<class_prob, gaussian_k, class_property_map>* m1 = new kml::svm<class_prob, gaussian_k, class_property_map>(*m);
    return (void *) m1;
      //    return (void *) new kml::svm<class_prob, gaussian_k, class_property_map>(*m);
  }

  void kml_delete_class_double_gaussian(void *v) {
    delete (kml::svm<class_prob, gaussian_k, class_property_map>*) v;
  }

  void kml_learn_class_double_gaussian(void *v, double **p, int *t,
						int sz_row, int sz_col) {
    kml::svm<class_prob, gaussian_k, class_property_map>* m = (kml::svm<class_prob, gaussian_k, class_property_map> *) v;
    boost::shared_ptr<class_property_map> data(new class_property_map);
    for (int j = 0; j < sz_row; ++j) {
      (*data)[j] = boost::make_tuple(std::vector<double>(*p, (*p) + sz_col), *t);
      ++p; ++t;
    }
    m->set_data(data);
    m->learn(data->storage_begin(), data->storage_end()); 
  }

  double kml_classify_double_gaussian(void *v, double *i, int sz) {
    kml::svm<class_prob, gaussian_k, class_property_map>* m = (kml::svm<class_prob, gaussian_k, class_property_map> *) v;
    if (m->operator()(std::vector<double>(i, i+sz)) > 0)
      return 1;
    else 
      return 0;
 // This way we can coerce to boolean trivially. Should try return (m->operator()(std::vector<double>(i, i+sz)) > 0) though.
  }

  void* kml_new_rank_double_gaussian(double k, double s) {
    boost::shared_ptr<rank_property_map> m(new rank_property_map);
    return (void *) new kml::svm<rank_prob, gaussian_k, rank_property_map>(k, s, m);
  }

  void* kml_copy_rank_double_gaussian(void *v) {
    kml::svm<rank_prob, gaussian_k, rank_property_map>* m = (kml::svm<rank_prob, gaussian_k, rank_property_map> *) v;
    return (void *) new kml::svm<rank_prob, gaussian_k, rank_property_map>(*m);
  }


  void kml_learn_rank_double_gaussian(void *v, double **p, int *g, int *t, 
					 int sz_row, int sz_col) {
    kml::svm<rank_prob, gaussian_k, rank_property_map>* m = (kml::svm<rank_prob, gaussian_k, rank_property_map> *) v;
    boost::shared_ptr<rank_property_map> data(new rank_property_map);
    for (int j = 0; j < sz_row; ++j) {
      (*data)[j] = boost::make_tuple(std::vector<double>(*p, (*p) + sz_col), *g, *t);
      ++p; ++g; ++t;
    }
    m->set_data(data);
    m->learn(data->storage_begin(), data->storage_end());
  }

  double kml_rank_double_gaussian(void* v, double *i, int sz) {
    kml::svm<rank_prob, gaussian_k, rank_property_map>* m = (kml::svm<rank_prob, gaussian_k, rank_property_map> *) v;
    return (m->operator()(std::vector<double>(i, i+sz)));
  }

  void* kml_new_class_double_polynomial(double g, double l, double d, double s) { 
    polynomial_k p(g, l, d);
    class_property_map m;
    return (void *) new kml::svm<class_prob, polynomial_k, class_property_map>(p, s, m);
  }

  void* kml_copy_class_double_polynomial(void *v) {
    kml::svm<class_prob, polynomial_k, class_property_map>* m = (kml::svm<class_prob, polynomial_k, class_property_map> *) v;
    return (void *) new kml::svm<class_prob, polynomial_k, class_property_map>(*m);
  }

  void kml_delete_class_double_polynomial(void *v) {
    delete (kml::svm<class_prob, polynomial_k, class_property_map>*) v;
  }

  void kml_learn_class_double_polynomial(void *v, double **p, int *t,
						int sz_row, int sz_col) {
    kml::svm<class_prob, polynomial_k, class_property_map>* m = (kml::svm<class_prob, polynomial_k, class_property_map> *) v;
    class_property_map data;
    for (int j = 0; j < sz_row; ++j) {
      data[j] = boost::make_tuple(std::vector<double>(*p, (*p) + sz_col), *t);
      ++p; ++t;
    }
    m->set_data(data);
    m->learn(data.storage_begin(), data.storage_end()); 
  }

  double kml_classify_double_polynomial(void *v, double *i, int sz) {
    kml::svm<class_prob, polynomial_k, class_property_map>* m = (kml::svm<class_prob, polynomial_k, class_property_map> *) v;
    if (m->operator()(std::vector<double>(i, i+sz)) > 0)
      return 1;
    else
      return 0; // This way we can coerce to boolean trivially. Should try return (m->operator()(std::vector<double>(i, i+sz)) > 0) though.
  }

  void* kml_new_rank_double_polynomial(double g, double l, double d, double s) {
    polynomial_k p(g, l, d);
    rank_property_map m;
    return (void *) new kml::svm<rank_prob, polynomial_k, rank_property_map>(p, s, m);
  }

  void* kml_copy_rank_double_polynomial(void *v) {
    kml::svm<rank_prob, polynomial_k, rank_property_map>* m = (kml::svm<rank_prob, polynomial_k, rank_property_map> *) v;
    return (void *) new kml::svm<rank_prob, polynomial_k, rank_property_map>(*m);
  }


  void kml_learn_rank_double_polynomial(void *v, double **p, int *g, int *t, 
					 int sz_row, int sz_col) {
    kml::svm<rank_prob, polynomial_k, rank_property_map>* m = (kml::svm<rank_prob, polynomial_k, rank_property_map> *) v;
    rank_property_map data;
    for (int j = 0; j < sz_row; ++j) {
      data[j] = boost::make_tuple(std::vector<double>(*p, (*p) + sz_col), *g, *t);
      ++p; ++g; ++t;
    }
    m->set_data(data);
    m->learn(data.storage_begin(), data.storage_end());
  }

  double kml_rank_double_polynomial(void* v, double *i, int sz) {
    kml::svm<rank_prob, polynomial_k, rank_property_map>* m = (kml::svm<rank_prob, polynomial_k, rank_property_map> *) v;
    return (m->operator()(std::vector<double>(i, i+sz)));
  }

  void* kml_new_class_double_linear(double s) { 
    class_property_map m;
    return (void *) new kml::svm<class_prob, linear_k, class_property_map>(linear_k(), s, m);
  }

  void* kml_copy_class_double_linear(void *v) {
    kml::svm<class_prob, linear_k, class_property_map>* m = (kml::svm<class_prob, linear_k, class_property_map> *) v;
    return (void *) new kml::svm<class_prob, linear_k, class_property_map>(*m);
  }

  void kml_delete_class_double_linear(void *v) {
    delete (kml::svm<class_prob, linear_k, class_property_map>*) v;
  }

  void kml_learn_class_double_linear(void *v, double **p, int *t,
						int sz_row, int sz_col) {
    kml::svm<class_prob, linear_k, class_property_map>* m = (kml::svm<class_prob, linear_k, class_property_map> *) v;
    class_property_map data;
    for (int j = 0; j < sz_row; ++j) {
      data[j] = boost::make_tuple(std::vector<double>(*p, (*p) + sz_col), *t);
      ++p; ++t;
    }
    m->set_data(data);
    m->learn(data.storage_begin(), data.storage_end()); 
  }

  double kml_classify_double_linear(void *v, double *i, int sz) {
    kml::svm<class_prob, linear_k, class_property_map>* m = (kml::svm<class_prob, linear_k, class_property_map> *) v;
    if (m->operator()(std::vector<double>(i, i+sz)) > 0)
      return 1;
    else
      return 0; // This way we can coerce to boolean trivially. Should try return (m->operator()(std::vector<double>(i, i+sz)) > 0) though.
  }

  void* kml_new_rank_double_linear(double s) {
    rank_property_map m;
    return (void *) new kml::svm<rank_prob, linear_k, rank_property_map>(linear_k(), s, m);
  }

  void* kml_copy_rank_double_linear(void *v) {
    kml::svm<rank_prob, linear_k, rank_property_map>* m = (kml::svm<rank_prob, linear_k, rank_property_map> *) v;
    return (void *) new kml::svm<rank_prob, linear_k, rank_property_map>(*m);
  }


  void kml_learn_rank_double_linear(void *v, double **p, int *g, int *t, 
					 int sz_row, int sz_col) {
    kml::svm<rank_prob, linear_k, rank_property_map>* m = (kml::svm<rank_prob, linear_k, rank_property_map> *) v;
    rank_property_map data;
    for (int j = 0; j < sz_row; ++j) {
      data[j] = boost::make_tuple(std::vector<double>(*p, (*p) + sz_col), *g, *t);
      ++p; ++g; ++t;
    }
    m->set_data(data);
    m->learn(data.storage_begin(), data.storage_end());
  }

  double kml_rank_double_linear(void* v, double *i, int sz) {
    kml::svm<rank_prob, linear_k, rank_property_map>* m = (kml::svm<rank_prob, linear_k, rank_property_map> *) v;
    return (m->operator()(std::vector<double>(i, i+sz)));
  }

}

#endif
