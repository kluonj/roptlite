#include "f2c.h"

namespace cwrapper {
#ifdef __cplusplus
extern "C" { 
#endif  



int dpbcon_(char *uplo, integer *n, integer *kd, doublereal *ab, integer *ldab, doublereal *anorm, doublereal *rcond, doublereal *work, integer *iwork, integer *info);

#ifdef __cplusplus
}
#endif
} // end of namespace cwrapper