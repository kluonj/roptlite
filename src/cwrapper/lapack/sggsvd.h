#pragma once

#include "f2c.h"

namespace cwrapper {
#ifdef __cplusplus
extern "C" { 
#endif  



int sggsvd_(char *jobu, char *jobv, char *jobq, integer *m, integer *n, integer *p, integer *k, integer *l, real *a, integer *lda, real *b, integer *ldb, real *alpha, real *beta, real *u, integer *ldu, real *v, integer *ldv, real *q, integer *ldq, real *work, integer *iwork, integer *info);

#ifdef __cplusplus
}
#endif
} // end of namespace cwrapper