include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(googletest)

add_executable(meshparam_tests
    test_laplacian.cpp
    test_heat_geodesics.cpp
    test_mds.cpp
    test_pipeline.cpp
    test_geodesic_accuracy.cpp
)

target_link_libraries(meshparam_tests PRIVATE
    meshparam
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(meshparam_tests)

# Debug tool for eigendecomposition analysis
add_executable(debug_eigen debug_eigen.cpp)
target_link_libraries(debug_eigen PRIVATE meshparam)
