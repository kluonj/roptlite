#pragma once

#include "f2c.h"

namespace cwrapper {
#ifdef __cplusplus
extern "C" { 
#endif  



int ztbrfs_(char *uplo, char *trans, char *diag, integer *n, integer *kd, integer *nrhs, doublecomplex *ab, integer *ldab, doublecomplex *b, integer *ldb, doublecomplex *x, integer *ldx, doublereal *ferr, doublereal *berr, doublecomplex *work, doublereal *rwork, integer *info);

#ifdef __cplusplus
}
#endif
} // end of namespace cwrapper