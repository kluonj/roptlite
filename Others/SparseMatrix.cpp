
#include "Others/SparseMatrix.h"

/*Define the namespace*/
namespace ROPTLITE{

    SparseMatrix::SparseMatrix(integer inm, integer inn, unsigned long *inir, unsigned long *injc, unsigned long *injcc, realdp *invals, integer innz)
    {
        m = inm;
        n = inn;
        iscomplex = false;
        ir = inir;
        jc = injc;
        jcc = injcc;
        valsreal = invals;
        valscomplex = nullptr;
        nz = innz;
        
//#ifdef SINGLE_PRECISION
//        SparseM = BLAS_suscr_begin(m, n);
//#else
//        SparseM = BLAS_duscr_begin(m, n);
//#endif
//        BLAS_uscr_insert_entries(SparseM, nz, vals, ir, jc);
//#ifdef SINGLE_PRECISION
//        BLAS_suscr_end(SparseM);
//#else
//        BLAS_duscr_end(SparseM);
//#endif
    };

    SparseMatrix::SparseMatrix(integer inm, integer inn, unsigned long *inir, unsigned long *injc, unsigned long *injcc, realdpcomplex *invals, integer innz)
    {
        m = inm;
        n = inn;
        iscomplex = true;
        ir = inir;
        jc = injc;
        jcc = injcc;
        valsreal = nullptr;
        valscomplex = invals;
        nz = innz;
        
//#ifdef SINGLE_PRECISION
//        SparseM = BLAS_cuscr_begin(m, n);
//#else
//        SparseM = BLAS_zuscr_begin(m, n);
//#endif
//        BLAS_uscr_insert_entries(SparseM, nz, vals, ir, jc);
//#ifdef SINGLE_PRECISION
//        BLAS_cuscr_end(SparseM);
//#else
//        BLAS_zuscr_end(SparseM);
//#endif
    };

    void SparseMatrix::Print(const char *name) const
    {
        printf("Print sparse matrix (row:%d, col:%d, nz:%d) %s:\n", m, n, nz, name);
        std::cout << "(";
        for(integer i = 0; i < n + 1; i++)
            std::cout << jcc[i] << ",";
        std::cout << ")" << std::endl;
        if(!iscomplex)
        {
            for(integer i = 0; i < nz; i++)
            {
                printf("i:%lu, j:%lu, val:%.10e\n", ir[i], jc[i], valsreal[i]);
            }
            return;
        }

        for(integer i = 0; i < nz; i++)
        {
            printf("i:%lu, j:%lu, val:%.10e+%.10e i\n", ir[i], jc[i], valscomplex[i].real(), valscomplex[i].imag());
        }
        if(nz == 0)
            printf("Empty matrix.\n");
        return;
    };

    SparseMatrix::~SparseMatrix(void)
    {
        if(ir != nullptr)
            delete [] ir;
        
        if(jc != nullptr)
            delete [] jc;
        
        if(jcc != nullptr)
            delete [] jcc;
        
        if(valsreal != nullptr)
            delete [] valsreal;
        
        if(valscomplex != nullptr)
            delete [] valscomplex;
    };

}; /*end of ROPTLITE namespace*/
