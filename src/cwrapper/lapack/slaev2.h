#pragma once

#include "f2c.h"

namespace cwrapper {
#ifdef __cplusplus
extern "C" { 
#endif  



int slaev2_(real *a, real *b, real *c__, real *rt1, real *rt2, real *cs1, real *sn1);

#ifdef __cplusplus
}
#endif
} // end of namespace cwrapper