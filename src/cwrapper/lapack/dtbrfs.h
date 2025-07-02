#include "f2c.h"

namespace cwrapper {
#ifdef __cplusplus
extern "C" { 
#endif  



int dtbrfs_(char *uplo, char *trans, char *diag, integer *n, integer *kd, integer *nrhs, doublereal *ab, integer *ldab, doublereal *b, integer *ldb, doublereal *x, integer *ldx, doublereal *ferr, doublereal *berr, doublereal *work, integer *iwork, integer *info);

#ifdef __cplusplus
}
#endif
} // end of namespace cwrapper