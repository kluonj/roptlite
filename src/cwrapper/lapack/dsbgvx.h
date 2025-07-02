#include "f2c.h"

namespace cwrapper {
#ifdef __cplusplus
extern "C" { 
#endif  



int dsbgvx_(char *jobz, char *range, char *uplo, integer *n, integer *ka, integer *kb, doublereal *ab, integer *ldab, doublereal *bb, integer *ldbb, doublereal *q, integer *ldq, doublereal *vl, doublereal *vu, integer *il, integer *iu, doublereal *abstol, integer *m, doublereal *w, doublereal *z__, integer *ldz, doublereal *work, integer *iwork, integer *ifail, integer *info);

#ifdef __cplusplus
}
#endif
} // end of namespace cwrapper