cmake_minimum_required(VERSION 3.0)

include(ExternalProject)

ExternalProject_Add(pcm
  PREFIX pcm
  GIT_REPOSITORY "git@github.com:opcm/pcm.git"
  GIT_TAG "202003"
  SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/pcm/src/pcm"
	CONFIGURE_COMMAND ""
	INSTALL_COMMAND ""
	BUILD_IN_SOURCE "${CMAKE_CURRENT_BINARY_DIR}/pcm/src/pcm"
	BUILD_COMMAND make lib
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/experiments/)

add_executable(pcm_pcie pcm_pcie/pcm_pcie.cpp)

#include_directories(${CMAKE_CURRENT_BINARY_DIR}/pcm/src)
target_include_directories(pcm_pcie PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/pcm/src)
target_include_directories(pcm_pcie PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(pcm_pcie
	deepplan
	${TORCH_LIBRARIES}
	${CMAKE_CURRENT_BINARY_DIR}/pcm/src/pcm/libPCM.a
)
