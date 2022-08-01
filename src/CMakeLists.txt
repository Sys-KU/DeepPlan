include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(SRC_LIST
		model.cpp
		engine.cpp
		util.cpp

		benchmark.cpp
)

set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

add_executable(benchmark ${SRC_LIST})
target_link_libraries(benchmark
	tbb
	deepplan_proto
	${Boost_LIBRARIES}
	${TORCH_LIBRARIES}
)

add_executable(test model.cpp engine.cpp util.cpp test.cpp)
target_link_libraries(test
	tbb
	deepplan_proto
	${Boost_LIBRARIES}
	${TORCH_LIBRARIES}
)
