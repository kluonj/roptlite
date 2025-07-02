#pragma once

#include "f2c.h"

namespace cwrapper {
#ifdef __cplusplus
extern "C" { 
#endif  



int zhbgst_(char *vect, char *uplo, integer *n, integer *ka, integer *kb, doublecomplex *ab, integer *ldab, doublecomplex *bb, integer *ldbb, doublecomplex *x, integer *ldx, doublecomplex *work, doublereal *rwork, integer *info);

#ifdef __cplusplus
}
#endif
} // end of namespace cwrapper