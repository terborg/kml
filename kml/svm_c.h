#ifdef __cplusplus
extern "C" {
#endif

  void* kml_new_classification_double_gaussian(double k, double s); /* need to instantiate with an empty data */
  void* kml_copy_classification_double_gaussian(void *v);
  void kml_delete_classification_double_gaussian(void *v);
  void kml_learn_classification_double_gaussian(void *v, double **p, int *t,
						int sz_row, int sz_col);
  double kml_classify_double_gaussian(void *v, double *i, int sz);
  
  void* kml_new_ranking_double_gaussian(double k, double s);
  void* kml_copy_ranking_double_gaussian(void *v);
  void kml_delete_ranking_double_gaussian(void *v);
  void kml_learn_ranking_double_gaussian(void *v, double **p, int *t, int *g,
					 int sz_row, int sz_col);
  double kml_rank_double_gaussian(void *v, double *i, int sz);

  void* kml_new_classification_double_polynomial(double k, double s);
  void* kml_copy_classification_double_polynomial(void *v);
  void kml_delete_classification_double_polynomial(void *v);
  void kml_learn_classification_double_polynomial(void *v, double **p, int *t,
						  int sz_row, int sz_col);
  double kml_classify_double_polynomial(void *v, double *i, int sz);
  
  void* kml_new_ranking_double_polynomial(double k, double s);
  void* kml_copy_ranking_double_polynomial(void *v);
  void kml_delete_ranking_double_polynomial(void *v);
  void kml_learn_ranking_double_polynomial(void *v, double **p, int *t, int *g,
					   int sz_row, int sz_col);
  double kml_rank_double_polynomial(void *v, double *i, int sz);
  
  void* kml_new_classification_double_linear(double k, double s);
  void* kml_copy_classification_double_linear(void *v);
  void kml_delete_classification_double_linear(void *v);
  void kml_learn_classification_double_linear(void *v, double **p, int *t,
					      int sz_row, int sz_col);
  double kml_classify_double_linear(void *v, double *i, int sz);

  void* kml_new_ranking_double_linear(double k, double s);
  void* kml_copy_ranking_double_linear(void *v);
  void kml_delete_ranking_double_linear(void *v);
  void kml_learn_ranking_double_linear(void *v, double **p, int *t, int *g,
				       int sz_row, int sz_col);
  double kml_rank_double_linear(void *v, double *i, int sz);

#ifdef __cplusplus
}
#endif
