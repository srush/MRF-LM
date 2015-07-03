#include <cmath>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#include <vector>

using namespace std;

// H(\mu_x)
double entropy(double *tab, int len) {
    double res = 0;
    for (int s = 0; s < len; s++) {
        if (not tab[s] == 0) {
            res -= tab[s] * log(tab[s]);
        }
    }
    return res;
}

// I(\mu_{x,y},\mu_x,\mu_y)
double mutual(double **array, double *tab1, double *tab2, int len) {
    double res = 0;
    double *auxres = new double[len];
    #pragma omp parallel for
    for (int s = 0; s < len; s++) {
        auxres[s] = 0;
        for (int t = 0; t < len; t++) {
            if (array[s][t] > 0) {
                auxres[s] += array[s][t]
                        * log(array[s][t] / (tab1[s] * tab2[t]));
            }
            if (array[s][t] < 0) {
                printf("\n BUG: negative marginal %.4e \n", array[s][t]);
            }
        }
    }
    for (int s = 0; s < len; s++) {
        res+=auxres[s];
    }
    delete[] auxres;
    return res;
}

//double logsumtab(double* t, int len) {
double logsumtab(vector<double> &t, int len) {
    double maxv = t[0];
    double result = 0;
    for (int i = 0; i < len; i++) {
        if (t[i] > maxv) {
            maxv = t[i];
        }
    }
    for (int i = 0; i < len; i++) {
        result+=exp(t[i]-maxv);
    }
     return maxv+log(result);
}

double logsumtab(float* t, int len) {
    double maxv = t[0];
    double result = 0;
    for (int i = 0; i < len; i++) {
        if (t[i] > maxv) {
            maxv = (double)t[i];
        }
    }
    for (int i = 0; i < len; i++) {
        result += exp((double)t[i] - maxv);
    }
    return maxv+log(result);
}

void exptab(vector<double> &tab, vector<double> &exptab, int len) {
    double maxt = tab[0];
    for (int s = 0; s < len; s++) {
        if (maxt < tab[s]) {
            maxt = tab[s];
        }
    }
    for (int s = 0; s < len; s++) {
        exptab[s] = exp(tab[s] - maxt);
    }
    exptab[len] = maxt;
}
