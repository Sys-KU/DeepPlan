cmake_minimum_required(VERSION 3.0)
#add_compile_options(-std=c++)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_C_STANDARD 11)

project(DeepPlan)

find_package(Torch REQUIRED)
find_package(Boost COMPONENTS system filesystem REQUIRED)

include_directories(${Boost_INCLUDE_DIR})

# Build protobuf v3.11.4
find_package(Protobuf REQUIRED)
include_directories(${Protobuf_INCLUDE_DIR})

include_directories(${CMAKE_CURRENT_BINARY_DIR})
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS proto/deepplan.proto)
add_library(deepplan_proto ${PROTO_HDRS} ${PROTO_SRCS})
target_link_libraries(deepplan_proto PRIVATE ${Protobuf_LIBRARIES})

protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS proto/deepcache.proto)
add_library(deepcache_proto ${PROTO_HDRS} ${PROTO_SRCS})
target_link_libraries(deepcache_proto PRIVATE ${Protobuf_LIBRARIES})

include_directories(${CMAKE_CURRENT_BINARY_DIR})
#protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS src/deepplan/proto/deepplan.proto)
#add_library(deepplan_proto ${PROTO_HDRS} ${PROTO_SRCS})
#target_link_libraries(deepplan_proto PRIVATE ${Protobuf_LIBRARIES})

add_subdirectory(src)
