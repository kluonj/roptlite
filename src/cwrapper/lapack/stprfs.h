#include "f2c.h"

namespace cwrapper {
#ifdef __cplusplus
extern "C" { 
#endif  



int stprfs_(char *uplo, char *trans, char *diag, integer *n, integer *nrhs, real *ap, real *b, integer *ldb, real *x, integer *ldx, real *ferr, real *berr, real *work, integer *iwork, integer *info);

#ifdef __cplusplus
}
#endif
} // end of namespace cwrapper