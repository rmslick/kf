project(kalmanfilter)
cmake_minimum_required(VERSION 3.0)
include(GNUInstallDirs)
find_package(autodiff REQUIRED)
find_package(Armadillo REQUIRED)
find_package(jsoncpp REQUIRED)
find_package(Eigen3 REQUIRED)
enable_testing()

set(SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/EKF.cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/LinearKF.cpp  
            ${CMAKE_CURRENT_SOURCE_DIR}/include/EKF.h ${CMAKE_CURRENT_SOURCE_DIR}/include/ExtendedKalmanFilter.h ${CMAKE_CURRENT_SOURCE_DIR}/include/LinearKF.h  ${CMAKE_CURRENT_SOURCE_DIR}/include/KF.h  )

add_executable(kalman_filter main.cpp ${SOURCES} )
set_target_properties(kalman_filter PROPERTIES CXX_STANDARD 17)

include_directories( ${autodiff_INCLUDE_DIRS} ${Armadillo_INCLUDE_DIRS} ${jsoncpp_INCLUDE_DIRS}  )
link_directories( ${autodiff_LIBRARY_DIRS} ${Armadillo_LIBRARY_DIRS} ${jsoncpp_LIBRARY_DIRS} )


# Link to teaserpp & Eigen3
target_include_directories(kalman_filter PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${autodiff_INCLUDE_DIRS} ${Armadillo_INCLUDE_DIRS} ${jsoncpp_INCLUDE_DIRS})
target_link_libraries(kalman_filter ${autodiff_LIBRARIES} armadillo  jsoncpp Eigen3::Eigen)
include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/include )
add_subdirectory(UnitTests)

# add library
add_library(kalmanfilter SHARED ${SOURCES})#${CMAKE_CURRENT_SOURCE_DIR}/src/EKF.cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/LinearKF.cpp)
set_target_properties(kalmanfilter PROPERTIES CXX_STANDARD 17)

target_link_libraries(kalmanfilter ${autodiff_LIBRARIES} armadillo  jsoncpp Eigen3::Eigen)

set_target_properties(kalmanfilter PROPERTIES SOVERSION 1)
target_include_directories(kalmanfilter PUBLIC include)
#target_include_directories(kalmanfilter PRIVATE src)

# For access to standard installation directory variables (CMAKE_INSTALL_xDIR).
message(${CMAKE_INSTALL_LIBDIR})
install(TARGETS kalmanfilter LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} )

# Create the pkg-config file from the template.
configure_file(${PROJECT_NAME}.pc.in ${PROJECT_NAME}.pc @ONLY)
# Set pkg-config file to install.
install(FILES ${CMAKE_BINARY_DIR}/${PROJECT_NAME}.pc
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
)
install (FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/EKF.h ${CMAKE_CURRENT_SOURCE_DIR}/include/LinearKF.h  ${CMAKE_CURRENT_SOURCE_DIR}/include/KF.h ${CMAKE_CURRENT_SOURCE_DIR}/include/AutoDiffWrapper.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})