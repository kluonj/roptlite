#include "f2c.h"

namespace cwrapper {
#ifdef __cplusplus
extern "C" { 
#endif  



int ctrevc_(char *side, char *howmny, logical *select, integer *n, complex *t, integer *ldt, complex *vl, integer *ldvl, complex *vr, integer *ldvr, integer *mm, integer *m, complex *work, real *rwork, integer *info);

#ifdef __cplusplus
}
#endif
} // end of namespace cwrapper