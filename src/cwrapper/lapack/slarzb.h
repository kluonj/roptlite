#pragma once

#include "f2c.h"

namespace cwrapper {
#ifdef __cplusplus
extern "C" { 
#endif  



int slarzb_(char *side, char *trans, char *direct, char *storev, integer *m, integer *n, integer *k, integer *l, real *v, integer *ldv, real *t, integer *ldt, real *c__, integer *ldc, real *work, integer *ldwork);

#ifdef __cplusplus
}
#endif
} // end of namespace cwrapper