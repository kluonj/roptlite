#pragma once

#include "f2c.h"

namespace cwrapper {
#ifdef __cplusplus
extern "C" { 
#endif  



int clarft_(char *direct, char *storev, integer *n, integer *k, complex *v, integer *ldv, complex *tau, complex *t, integer *ldt);

#ifdef __cplusplus
}
#endif
} // end of namespace cwrapper