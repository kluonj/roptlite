#pragma once

#include "f2c.h"

namespace cwrapper {
#ifdef __cplusplus
extern "C" { 
#endif  



int dlags2_(logical *upper, doublereal *a1, doublereal *a2, doublereal *a3, doublereal *b1, doublereal *b2, doublereal *b3, doublereal *csu, doublereal *snu, doublereal *csv, doublereal *snv, doublereal *csq, doublereal *snq);

#ifdef __cplusplus
}
#endif
} // end of namespace cwrapper