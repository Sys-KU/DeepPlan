add_executable(load_interference load_interference.cpp)
target_link_libraries(load_interference
	deepplan
	${Boost_LIBRARIES}
	${TORCH_LIBRARIES}
)

