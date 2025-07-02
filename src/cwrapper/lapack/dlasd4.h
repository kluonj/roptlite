#pragma once

#include "f2c.h"

namespace cwrapper {
#ifdef __cplusplus
extern "C" { 
#endif  



int dlasd4_(integer *n, integer *i__, doublereal *d__, doublereal *z__, doublereal *delta, doublereal *rho, doublereal *sigma, doublereal *work, integer *info);

#ifdef __cplusplus
}
#endif
} // end of namespace cwrapper