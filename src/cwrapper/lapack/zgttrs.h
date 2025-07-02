#pragma once

#include "f2c.h"

namespace cwrapper {
#ifdef __cplusplus
extern "C" { 
#endif  



int zgttrs_(char *trans, integer *n, integer *nrhs, doublecomplex *dl, doublecomplex *d__, doublecomplex *du, doublecomplex *du2, integer *ipiv, doublecomplex *b, integer *ldb, integer *info);

#ifdef __cplusplus
}
#endif
} // end of namespace cwrapper