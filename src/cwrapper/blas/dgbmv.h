#include "f2c.h" 

namespace cwrapper {
#ifdef __cplusplus
extern "C" { 
#endif  




int dgbmv_(char *trans, integer *m, integer *n, integer *kl, integer *ku, doublereal *alpha, doublereal *a, integer *lda, doublereal *x, integer *incx, doublereal *beta, doublereal *y, integer *incy);

#ifdef __cplusplus
}
#endif
} // end of namespace cwrapper