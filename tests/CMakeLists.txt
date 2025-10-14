set(ROPTLITE_TEST_SOURCES
  DriverCpp.cpp
  DriverJuliaProb.cpp
  DriverMexProb.cpp
  TestCFRankQ2FBlindDecon2D.cpp
  TestCSFRQPhaseRetrieval.cpp
  TestCStieBrockett.cpp
  TestElement.cpp
  TestEucQuadratic.cpp
  TestFRankE3FMatCompletion.cpp
  TestFRankESparseApprox.cpp
  TestFRankETextureInpainting.cpp
  TestFRankEWeightApprox.cpp
  TestFRankQ2FMatCompletion.cpp
  TestGrassMatCompletion.cpp
  TestGrassPCA.cpp
  TestGrassRQ.cpp
  TestGrassSVPCA.cpp
  TestObliqueSPCA.cpp
  TestPoincareEmbeddings.cpp
  TestProdStieSumBrockett.cpp
  TestSFRQLyapunov.cpp
  TestSPDKarcherMean.cpp
  TestSphereSparsestVector.cpp
  TestStieBrockett.cpp
  TestStieSPCA.cpp
  TestStieSoftICA.cpp
)

add_executable(roptlite_tests ${ROPTLITE_TEST_SOURCES})

target_include_directories(roptlite_tests PRIVATE
  ${PROJECT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(roptlite_tests PRIVATE ROPTLITE)

add_test(NAME roptlite_tests COMMAND roptlite_tests)
