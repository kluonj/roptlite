#pragma once

#include "f2c.h"

namespace cwrapper {
#ifdef __cplusplus
extern "C" { 
#endif  



int dlasd0_(integer *n, integer *sqre, doublereal *d__, doublereal *e, doublereal *u, integer *ldu, doublereal *vt, integer *ldvt, integer *smlsiz, integer *iwork, doublereal *work, integer *info);

#ifdef __cplusplus
}
#endif
} // end of namespace cwrapper