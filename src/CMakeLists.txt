include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(DEEPPLAN_SRC
		util.cpp
		deepplan/model.cpp
		deepplan/engine.cpp
		libtorch/model.cpp
)

set(DEEPCACHE_SRC
		util.cpp
		deepcache/model.cpp
		deepcache/engine.cpp
		libtorch/model.cpp
)

set(NETWORK_SRC
		network/network.cpp
		network/session.cpp
		network/server_api.cpp)


set(SERVER_SRC
		server.cpp
		server/server.cpp
		server/controller.cpp
		server/model_manager.cpp
		server/worker.cpp
)

set(CLIENT_SRC
		client.cpp
		client/genzipf.cpp
		client/client.cpp
		client/workload.cpp
)

set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

add_library(deepplan ${DEEPPLAN_SRC})
target_link_libraries(deepplan PRIVATE
	deepplan_proto
	tbb
	${Boost_LIBRARIES}
	${TORCH_LIBRARIES}
)

add_library(deepcache ${DEEPCACHE_SRC})
target_link_libraries(deepcache PRIVATE
	deepplan_proto
	tbb
	${Boost_LIBRARIES}
	${TORCH_LIBRARIES}
)

add_library(network ${NETWORK_SRC})
target_link_libraries(network PRIVATE
	deepplan_proto
	deepcache_proto
	tbb
	${Boost_LIBRARIES}
)

add_executable(benchmark benchmark.cpp)
target_link_libraries(benchmark
	deepplan
	${Boost_LIBRARIES}
	${TORCH_LIBRARIES}
)

add_executable(server ${SERVER_SRC})
target_link_libraries(server
	deepplan
	deepcache
	network
	tbb
	${Boost_LIBRARIES}
	${TORCH_LIBRARIES}
)

add_executable(client ${CLIENT_SRC})
target_link_libraries(client
	deepplan
	network
	tbb
	${Boost_LIBRARIES}
	${TORCH_LIBRARIES}
)

add_executable(cache_study cache_study.cpp)
target_link_libraries(cache_study
	deepcache
	${Boost_LIBRARIES}
	${TORCH_LIBRARIES}
)

add_subdirectory(experiments)

