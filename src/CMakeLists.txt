include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(SRC_LIST
		util.cpp

		core/model.cpp
		core/engine.cpp

		network/network.cpp
		network/session.cpp
		network/server_api.cpp
)

set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

add_library(deepplan ${SRC_LIST})
target_link_libraries(deepplan PRIVATE
	deepplan_proto
	deepcache_proto
	tbb
	${Boost_LIBRARIES}
	${TORCH_LIBRARIES})

add_executable(benchmark benchmark.cpp)
target_link_libraries(benchmark
	deepplan
	${Boost_LIBRARIES}
	${TORCH_LIBRARIES}
)

add_executable(server server.cpp controller.cpp)
target_link_libraries(server
	deepplan
	${Boost_LIBRARIES}
	${TORCH_LIBRARIES}
)

add_executable(client client.cpp)
target_link_libraries(client
	deepplan
	${Boost_LIBRARIES}
	${TORCH_LIBRARIES}
)


