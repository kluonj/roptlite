#include "f2c.h"

namespace cwrapper {
#ifdef __cplusplus
extern "C" { 
#endif  



int slacpy_(char *uplo, integer *m, integer *n, real *a, integer *lda, real *b, integer *ldb);

#ifdef __cplusplus
}
#endif
} // end of namespace cwrapper