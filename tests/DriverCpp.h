/*
This is the test file to check all problems.

---- WH
*/

#ifndef DRIVERCPP_H
#define DRIVERCPP_H

#include "Others/def.h"

//#include "tests/TestCFR2BlindDecon2D.h"
//#include "tests/TestCFR2BlindDeconvolution.h"
//#include "tests/TestCSO.h"
//#include "tests/TestCSOPhaseRetrieval.h"
//#include "tests/TestElasticCurvesRO.h"
//#include "tests/TestEucBlindDeconvolution.h"
//#include "tests/TestEucFrechetMean.h"
//#include "tests/TestEucPosSpCd.h"
#include "tests/TestCFRankQ2FBlindDecon2D.h"
#include "tests/TestCSFRQPhaseRetrieval.h"
#include "tests/TestCStieBrockett.h"
#include "tests/TestElement.h"
#include "tests/TestEucQuadratic.h"
#include "tests/TestFRankE3FMatCompletion.h" /*example of sparse matrix operations*/
#include "tests/TestFRankESparseApprox.h" /* example of proximal gradient method on the fixed-rank matrix manifold */
#include "tests/TestFRankETextureInpainting.h" /* example of proximal gradient method on the fixed-rank matrix manifold */
#include "tests/TestFRankEWeightApprox.h"
#include "tests/TestFRankQ2FMatCompletion.h" /*example of sparse matrix operations*/
#include "tests/TestGrassMatCompletion.h"
#include "tests/TestGrassPCA.h"
#include "tests/TestGrassRQ.h"
#include "tests/TestGrassSVPCA.h"
//#include "tests/TestKarcherMean.h"
//#include "tests/TestLRBlindDeconvolution.h"
//#include "tests/TestLRMatrixCompletion.h"
//#include "tests/TestLRSparseMatrixApprox.h"
//#include "tests/TestMNCRModel.h"
//#include "tests/TestMyMatrix.h"
//#include "tests/TestNSOLyapunov.h"
//#include "tests/TestObliqueQCor.h"
#include "tests/TestObliqueSPCA.h"
//#include "tests/TestOrthBoundingBox.h"
//#include "tests/TestPreShapePathStraighten.h"
#include "tests/TestPoincareEmbeddings.h"
//#include "tests/TestProduct.h"
#include "tests/TestProdStieSumBrockett.h"
#include "tests/TestSFRQLyapunov.h"
//#include "tests/TestShapePathStraighten.h"
//#include "tests/TestSparsePCA.h"
#include "tests/TestSPDKarcherMean.h"
//#include "tests/TestSPDMeanL1.h"
//#include "tests/TestSPDMeanLinfty.h"
//#include "tests/TestSPDMeanLDOneParam.h"
//#include "tests/TestSPDMeanLDOneParamL1.h"
//#include "tests/TestSPDMeanLDOneParamLinfty.h"
//#include "tests/TestSPDMeanSymmLDOneParam.h"
//#include "tests/TestSPDMeanSymmLDOneParamL1.h"
//#include "tests/TestSPDMeanSymmLDOneParamLinfty.h"
//#include "tests/TestSPDMeanMajorization.h"
//#include "tests/TestSPDTensorDL.h"
//#include "tests/TestSphereRayQuo.h"
#include "tests/TestSphereSparsestVector.h" /* example of subgradient-based methods for nonsmooth optimization*/
#include "tests/TestStieBrockett.h"
#include "tests/TestStieSoftICA.h"
//#include "tests/TestStieSparseBrockett.h"
#include "tests/TestStieSPCA.h" /* example of proximal gradient method on the Stiefel manifold */
//#include "tests/TestTestSparsePCA.h"
//#include "tests/TestWeightedLowRank.h"

//#include "Problems/CSFRQPhaseRetrieval.h"

using namespace ROPTLITE;

void testall(void);
/*numGradHess: 0, test both given gradient/Hessian and numerical gradient/Hessian
               1, test only given gradient/Hessian
               2, test only numerical gradient/Hessian */
void testSmoothProblem(Problem *prob, Variable *initx, const char *probname, std::vector<std::string> Methodnames, integer numGradHess = 0);
void testStoSmoothProblem(Problem *Prob, Variable *initx, const char *probname, std::vector<std::string> Methodnames, realdp initstepsizes[], integer numGradHess = 0);
void testProxGradProblem(Problem *prob, Variable *initx, const char *probname, std::vector<std::string> Methodnames);
void testSubGradProblem(Problem *prob, Variable *initx, const char *probname, std::vector<std::string> Methodnames);
bool stringinclude(std::vector<std::string> names, std::string name);
int main(void);



#endif
