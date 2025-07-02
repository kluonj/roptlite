#pragma once

#include "f2c.h"

namespace cwrapper {
#ifdef __cplusplus
extern "C" { 
#endif  



int zstedc_(char *compz, integer *n, doublereal *d__, doublereal *e, doublecomplex *z__, integer *ldz, doublecomplex *work, integer *lwork, doublereal *rwork, integer *lrwork, integer *iwork, integer *liwork, integer *info);

#ifdef __cplusplus
}
#endif
} // end of namespace cwrapper