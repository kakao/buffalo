cmake_minimum_required (VERSION 2.8.8)
project (cbuffalo)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -fPIC -Wall -Wextra -fopenmp -O3 -ggdb -march=native")

include(GNUInstallDirs)

set(SOURCES
    "./3rd/json11/json11.cpp"
)
file(GLOB ALGO_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/lib/*.cc)
file(GLOB MISC_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/lib/misc/*.cc)
file(GLOB ALS_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/lib/algo_impl/als/*.cc)
file(GLOB CFR_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/lib/algo_impl/cfr/*.cc)
file(GLOB BPR_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/lib/algo_impl/bpr/*.cc)
file(GLOB WARP_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/lib/algo_impl/warp/*.cc)
file(GLOB W2V_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/lib/algo_impl/w2v/*.cc)
file(GLOB PLSI_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/lib/algo_impl/plsi/*.cc)
add_library(cbuffalo SHARED ${SOURCES} ${ALGO_SRCS} ${ALS_SRCS} ${CFR_SRCS} ${BPR_SRCS} ${WARP_SRCS} ${W2V_SRCS} ${PLSI_SRCS} ${MISC_SRCS})
target_include_directories(cbuffalo
PRIVATE
    ./include
    ./3rd/json11
    ./3rd/spdlog/include
    ./3rd/n2/include
    ./3rd/eigen3
)

set_target_properties(cbuffalo PROPERTIES VERSION 0.1.0)
set_target_properties(cbuffalo PROPERTIES SOVERSION 1)

install(TARGETS cbuffalo
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(DIRECTORY "${CMAKE_SOURCE_DIR}/include/buffalo" # source directory
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}" # target directory
        FILES_MATCHING # install only matched files
        PATTERN "*.hpp" # select header files
)

set(N2_DIR "./3rd/n2")
file(GLOB_RECURSE N2_SRC CONFIGURE_DEPENDS ${N2_DIR}/src/*.cc)
add_library(n2 SHARED ${N2_SRC})
target_compile_options(n2 PRIVATE
    ${OpenMP_CXX_FLAGS}
    "-DBOOST_DISABLE_ASSERTS"
)
target_link_libraries(n2 PRIVATE pthread)
target_include_directories(n2
PRIVATE
    ${N2_DIR}/include/
    ${N2_DIR}/third_party/eigen/
    ${N2_DIR}/third_party/spdlog/include/
    ${N2_DIR}/third_party/boost/mpl/include/
    ${N2_DIR}/third_party/boost/bind/include/
    ${N2_DIR}/third_party/boost/core/include/
    ${N2_DIR}/third_party/boost/heap/include/
    ${N2_DIR}/third_party/boost/mp11/include/
    ${N2_DIR}/third_party/boost/assert/include/
    ${N2_DIR}/third_party/boost/config/include/
    ${N2_DIR}/third_party/boost/detail/include/
    ${N2_DIR}/third_party/boost/utility/include/
    ${N2_DIR}/third_party/boost/iterator/include/
    ${N2_DIR}/third_party/boost/parameter/include/
    ${N2_DIR}/third_party/boost/type_traits/include/
    ${N2_DIR}/third_party/boost/preprocessor/include/
    ${N2_DIR}/third_party/boost/concept_check/include/
    ${N2_DIR}/third_party/boost/static_assert/include/
    ${N2_DIR}/third_party/boost/throw_exception/include/
)
