#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>

extern "C" {
  void R_init_diffuR(DllInfo *dll){
    // nothing special; Rcpp will register symbols automatically
  }
}
