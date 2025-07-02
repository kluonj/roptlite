#pragma once

#include "f2c.h"

namespace cwrapper {
#ifdef __cplusplus
extern "C" { 
#endif  



int dlasd6_(integer *icompq, integer *nl, integer *nr, integer *sqre, doublereal *d__, doublereal *vf, doublereal *vl, doublereal *alpha, doublereal *beta, integer *idxq, integer *perm, integer *givptr, integer *givcol, integer *ldgcol, doublereal *givnum, integer *ldgnum, doublereal *poles, doublereal *difl, doublereal *difr, doublereal *z__, integer *k, doublereal *c__, doublereal *s, doublereal *work, integer *iwork, integer *info);

#ifdef __cplusplus
}
#endif
} // end of namespace cwrapper