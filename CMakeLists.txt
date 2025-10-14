# PROJECT CONFIGURATION
cmake_minimum_required(VERSION 3.10)

project(ROPTLITE  LANGUAGES CXX)

set (CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")
add_compile_options(-march=native) # Enable faster instruction sets

# Build all binaries in a separate directory
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)


include(GNUInstallDirs)
include(CTest)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
set(ROPTLITE_INSTALL_INCLUDEDIR ${CMAKE_INSTALL_INCLUDEDIR}/ROPTLITE)

# # Directory for built libraries
# set(LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR}/lib CACHE PATH "The directory in which to place libraries built by this project")
# # Directory for built executables
# set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR}/bin CACHE PATH "The directory in which to place executables built by this project")

# # Directory for built libraries
# set(LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR}/lib CACHE PATH "The directory in which to place libraries built by this project")
# # Directory for built executables
# set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR}/bin CACHE PATH "The directory in which to place executables built by this project")

# BUILD CONFIGURATIONS
option(CMAKE_VERBOSE_MAKEFILE "Generate verbose makefiles?" OFF)

set(CODE_PROFILING OFF CACHE BOOL "Turn on code profiling?")
if(${CODE_PROFILING})
message(STATUS "Turning on code profiling for ROPTLITE")
endif()


# ROPTLITE requires lapack, blas, and m libraries
find_library(LAPACK lapack DOC "LAPACK library to link against")
find_library(BLAS blas DOC "BLAS library to link against")
find_library(M m "M library to link against")

set(SRC_LIST

Manifolds/CFixedRankQ2F.cpp
Manifolds/CStiefel.cpp
Manifolds/CSymFixedRankQ.cpp
Manifolds/Element.cpp
Manifolds/Euclidean.cpp
Manifolds/FixedRankE.cpp
Manifolds/FixedRankE3F.cpp
Manifolds/FixedRankQ2F.cpp
Manifolds/Grassmann.cpp
Manifolds/Manifold.cpp
Manifolds/MultiManifolds.cpp
Manifolds/PoincareBall.cpp
Manifolds/SPDManifold.cpp
Manifolds/SmartSpace.cpp
Manifolds/Sphere.cpp
Manifolds/SphereTx.cpp
Manifolds/Stiefel.cpp
Manifolds/SymFixedRankQ.cpp

Others/BlasLapackCppWrapper.cpp
Others/ForDebug.cpp
Others/MinPNormConHull.cpp
Others/SparseMatrix.cpp
Others/Spline.cpp
Others/Timer.cpp
Others/randgen.cpp

Others/SparseBLAS/nist_spblas.cpp

Others/wavelet/wavelet.cpp

Problems/CFRankQ2FBlindDecon2D.cpp
Problems/CSFRQPhaseRetrieval.cpp
Problems/CStieBrockett.cpp
Problems/EucQuadratic.cpp
Problems/FRankE3FMatCompletion.cpp
Problems/FRankESparseApprox.cpp
Problems/FRankETextureInpainting.cpp
Problems/FRankEWeightApprox.cpp
Problems/FRankQ2FMatCompletion.cpp
Problems/GrassMatCompletion.cpp
Problems/GrassPCA.cpp
Problems/GrassRQ.cpp
Problems/GrassSVPCA.cpp
Problems/ObliqueSPCA.cpp
Problems/PoincareEmbeddings.cpp
Problems/Problem.cpp
Problems/ProdStieSumBrockett.cpp
Problems/SFRQLyapunov.cpp
Problems/SPDKarcherMean.cpp
Problems/SphereConvexHull.cpp
Problems/SphereSparsestVector.cpp
Problems/SphereTxRQ.cpp
Problems/StieBrockett.cpp
Problems/StieSPCA.cpp
Problems/StieSoftICA.cpp
Problems/juliaProblem.cpp
Problems/mexProblem.cpp

Solvers/IARPG.cpp
Solvers/IRPG.cpp
Solvers/LRBFGS.cpp
Solvers/LRBFGSSub.cpp
Solvers/LRBroydenFamily.cpp
Solvers/LRTRSR1.cpp
Solvers/LRTRSR1woR.cpp
Solvers/RADAM.cpp
Solvers/RADAMSP.cpp
Solvers/RAMSGRAD.cpp
Solvers/RAMSGRADSP.cpp
Solvers/RBFGS.cpp
Solvers/RBFGSSub.cpp
Solvers/RBroydenFamily.cpp
Solvers/RCG.cpp
Solvers/RGS.cpp
Solvers/RNewton.cpp
Solvers/RSD.cpp
Solvers/RSGD.cpp
Solvers/RSVRG.cpp
Solvers/RTRNewton.cpp
Solvers/RTRSD.cpp
Solvers/RTRSR1.cpp
Solvers/RWRBFGS.cpp
Solvers/SVRLRBFGS.cpp
Solvers/SVRLRBroydenFamily.cpp
Solvers/Solvers.cpp
Solvers/SolversNSM.cpp
Solvers/SolversNSMPGLS.cpp
Solvers/SolversNSMSub.cpp
Solvers/SolversNSMSubLS.cpp
Solvers/SolversSM.cpp
Solvers/SolversSMLS.cpp
Solvers/SolversSMSVRG.cpp
Solvers/SolversSMSto.cpp
Solvers/SolversSMTR.cpp
)


# COMPILE THE ROPTLITE LIBRARY
add_library(ROPTLITE SHARED ${SRC_LIST})

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

target_include_directories(ROPTLITE PUBLIC 
   PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Manifolds>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Others>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Others/SparseBLAS>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Others/fftw>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Others/wavelet>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Problems>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Solvers>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cwrapper>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cwrapper/blas>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cwrapper/lapack>

    $<INSTALL_INTERFACE:${ROPTLITE_INSTALL_INCLUDEDIR}>
    $<INSTALL_INTERFACE:${ROPTLITE_INSTALL_INCLUDEDIR}/Manifolds>
    $<INSTALL_INTERFACE:${ROPTLITE_INSTALL_INCLUDEDIR}/Others>
    $<INSTALL_INTERFACE:${ROPTLITE_INSTALL_INCLUDEDIR}/Others/SparseBLAS>
    $<INSTALL_INTERFACE:${ROPTLITE_INSTALL_INCLUDEDIR}/Others/fftw>
    $<INSTALL_INTERFACE:${ROPTLITE_INSTALL_INCLUDEDIR}/Others/wavelet>
    $<INSTALL_INTERFACE:${ROPTLITE_INSTALL_INCLUDEDIR}/Problems>
    $<INSTALL_INTERFACE:${ROPTLITE_INSTALL_INCLUDEDIR}/Solvers>
    $<INSTALL_INTERFACE:${ROPTLITE_INSTALL_INCLUDEDIR}/cwrapper>
    $<INSTALL_INTERFACE:${ROPTLITE_INSTALL_INCLUDEDIR}/cwrapper/blas>
    $<INSTALL_INTERFACE:${ROPTLITE_INSTALL_INCLUDEDIR}/cwrapper/lapack>
   
)

target_link_libraries(ROPTLITE ${LAPACK} ${BLAS} ${M})
set_target_properties(ROPTLITE PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)

if(${CODE_PROFILING})
	set_target_properties(ROPTLITE PROPERTIES COMPILE_FLAGS "-pg -g" LINK_FLAGS "-pg -g")
endif()

# COMPILE EXECUTABLES

# Compile the simple test program from ROPTLITE
# add_executable(TestSimpleExample test/TestSimpleExample.cpp)
# target_link_libraries(TestSimpleExample ROPTLITE)

# Compile the simple product test program from ROPTLITE
# add_executable(TestProduct test/TestProduct.h test/TestProduct.cpp)
# target_link_libraries(TestProduct ROPTLITE)

# Compile the simple product test program from ROPTLITE
# add_executable(TestProductBrockett test/TestProductExample.cpp test/TestProductExample.cpp)
# target_link_libraries(TestProductBrockett ROPTLITE)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()

################################ INSTALL ##########################################
# include(CMakePackageConfigHelpers)
# set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/ROPTLITE)

# write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/ROPTLITEConfigVersion.cmake
#   VERSION 0.7
#   COMPATIBILITY AnyNewerVersion
# )

# configure_package_config_file(${CMAKE_CURRENT_LIST_DIR}/cmake/ROPTLITEConfig.cmake.in
#   ${CMAKE_CURRENT_BINARY_DIR}/ROPTLITEConfig.cmake
#   INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
# )

# install the configuration file
# install(FILES
#   ${CMAKE_CURRENT_BINARY_DIR}/ROPTLITEConfig.cmake
#   DESTINATION ${INSTALL_CONFIGDIR}
# )


# install(
# 		TARGETS ROPTLITE
# 		EXPORT ROPTLITETargets
# 		LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
# )

# install(
# 		EXPORT ROPTLITETargets
# 		FILE ROPTLITETargets.cmake
# 		DESTINATION ${INSTALL_CONFIGDIR}
# )

# # install header files
# install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
# 		DESTINATION ${ROPTLITE_INSTALL_INCLUDEDIR}
# 		FILES_MATCHING PATTERN "*.h"
# 		)


# export(TARGETS ROPTLITE FILE ROPTLITETargets.cmake)

# ############################### UNINSTALL ##########################################
# if(NOT TARGET uninstall)
# 	configure_file(
# 			"${CMAKE_CURRENT_SOURCE_DIR}/cmake/cmake_uninstall.cmake.in"
# 			"${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
# 			IMMEDIATE @ONLY)

# 	add_custom_target(uninstall
# 			COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
# endif()
