#ifndef SVM_C_H
#define SVM_C_H

#ifdef __cplusplus
extern "C" {
#endif

  void* kml_new_class_double_gaussian(double k, double s); 
  void* kml_copy_class_double_gaussian(void *v);
  void kml_delete_class_double_gaussian(void *v);
  void kml_learn_class_double_gaussian(void *v, double **p, int *t,
						int sz_row, int sz_col);
  double kml_classify_double_gaussian(void *v, double *i, int sz);
  
  void* kml_new_rank_double_gaussian(double k, double s);
  void* kml_copy_rank_double_gaussian(void *v);
  void kml_delete_rank_double_gaussian(void *v);
  void kml_learn_rank_double_gaussian(void *v, double **p, int *g, int *t,
					 int sz_row, int sz_col);
  double kml_rank_double_gaussian(void *v, double *i, int sz);

  void* kml_new_class_double_polynomial(double g, double l, double d, double s);
  void* kml_copy_class_double_polynomial(void *v);
  void kml_delete_class_double_polynomial(void *v);
  void kml_learn_class_double_polynomial(void *v, double **p, int *t,
						  int sz_row, int sz_col);
  double kml_classify_double_polynomial(void *v, double *i, int sz);
  
  void* kml_new_rank_double_polynomial(double g, double l, double d, double s);
  void* kml_copy_rank_double_polynomial(void *v);
  void kml_delete_rank_double_polynomial(void *v);
  void kml_learn_rank_double_polynomial(void *v, double **p, int *t, int *g,
					   int sz_row, int sz_col);
  double kml_rank_double_polynomial(void *v, double *i, int sz);
  

  void* kml_new_class_double_linear(double k, double s);
  void* kml_copy_class_double_linear(void *v);
  void kml_delete_class_double_linear(void *v);
  void kml_learn_class_double_linear(void *v, double **p, int *t,
					      int sz_row, int sz_col);
  double kml_classify_double_linear(void *v, double *i, int sz);

  void* kml_new_rank_double_linear(double k, double s);
  void* kml_copy_rank_double_linear(void *v);
  void kml_delete_rank_double_linear(void *v);
  void kml_learn_rank_double_linear(void *v, double **p, int *t, int *g,
				       int sz_row, int sz_col);
  double kml_rank_double_linear(void *v, double *i, int sz);

#ifdef __cplusplus
}
#endif

#endif // SVM_C_H
