project(tests)
cmake_minimum_required(VERSION 3.0)
 
# Locate GTest
find_package(GTest REQUIRED)
find_package(Eigen3 REQUIRED)

enable_testing()

include_directories(${GTEST_INCLUDE_DIRS})
#set(SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/EKF.cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/LinearKF.cpp  
#${CMAKE_CURRENT_SOURCE_DIR}/include/EKF.h ${CMAKE_CURRENT_SOURCE_DIR}/include/LinearKF.h  ${CMAKE_CURRENT_SOURCE_DIR}/include/KF.h  )

# Link runTests with what we want to test and the GTest and pthread library
add_executable(runTests test.cpp)# ${SOURCES})

set_target_properties(runTests PROPERTIES CXX_STANDARD 17)

target_link_libraries(runTests ${GTEST_LIBRARIES} pthread  Eigen3::Eigen)

#add_test(multiply_gtests multiply_test)
# make test: https://stackoverflow.com/questions/48817444/cmake-subdirectories-and-google-test-mock-not-linking-properly-tests-not-fou
add_test(AutoDiffTest ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/runTests)
