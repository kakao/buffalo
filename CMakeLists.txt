cmake_minimum_required (VERSION 2.8.8)
project (cbuffalo)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -fPIC -Wall -Wextra -fopenmp -O3 -ggdb -march=native")

include(GNUInstallDirs)

include_directories("./include")
include_directories("./3rd/json11")
include_directories("./3rd/spdlog/include")
include_directories("./3rd/n2/include")
include_directories("./3rd/eigen3")

set(SOURCES
    "./3rd/json11/json11.cpp"
)
file(GLOB ALGO_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/lib/*.cc)
file(GLOB MISC_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/lib/misc/*.cc)
file(GLOB ALS_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/lib/algo_impl/als/*.cc)
file(GLOB CFR_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/lib/algo_impl/cfr/*.cc)
file(GLOB BPR_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/lib/algo_impl/bpr/*.cc)
file(GLOB W2V_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/lib/algo_impl/w2v/*.cc)
add_library(cbuffalo SHARED ${SOURCES} ${ALGO_SRCS} ${ALS_SRCS} ${CFR_SRCS} ${BPR_SRCS} ${W2V_SRCS} ${MISC_SRCS})
include_directories(cbuffalo "/usr/local/include/eigen3")

set_target_properties(cbuffalo PROPERTIES VERSION 0.1.0)
set_target_properties(cbuffalo PROPERTIES SOVERSION 1)

install(TARGETS cbuffalo
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(DIRECTORY "${CMAKE_SOURCE_DIR}/include/buffalo" # source directory
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}" # target directory
        FILES_MATCHING # install only matched files
        PATTERN "*.hpp" # select header files
)
