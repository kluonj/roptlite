#pragma once

#include "f2c.h"

namespace cwrapper {
#ifdef __cplusplus
extern "C" { 
#endif  



int zgbtrf_(integer *m, integer *n, integer *kl, integer *ku, doublecomplex *ab, integer *ldab, integer *ipiv, integer *info);

#ifdef __cplusplus
}
#endif
} // end of namespace cwrapper