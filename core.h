#ifndef CORE_H
#define CORE_H


double entropy(double *tab, int len);
double mutual(double **array, double *tab1, double *tab2, int len);
double logsumtab(float* t, int len);
double logsumtab(vector<double> &t, int len);

/* void exptab(double *tab, double *exptab, int len); */
void exptab(vector<double> &tab, vector<double> &exptab, int len);

#endif
