FIND_PATH(NUMA_INCLUDE_DIR NAME numa.h
  HINTS $ENV{HOME}/local/include /opt/local/include /usr/local/include /usr/include)

FIND_LIBRARY(NUMA_LIBRARIES NAME numa
  HINTS $ENV{HOME}/local/lib64 $ENV{HOME}/local/lib /usr/local/lib64 /usr/local/lib /opt/local/lib64 /opt/local/lib /usr/lib64 /usr/lib
)

IF (NUMA_INCLUDE_DIR AND NUMA_LIBRARIES)
    SET(NUMA_FOUND TRUE)
    MESSAGE(STATUS "Found numa library: inc=${NUMA_INCLUDE_DIR}, lib=${NUMA_LIBRARIES}")
ELSE ()
    SET(NUMA_FOUND FALSE)
    MESSAGE(STATUS "WARNING: Numa library not found.")
    MESSAGE(STATUS "Try: 'sudo apt-get install libnuma libnuma-dev' (or sudo yum install numactl numactl-devel)")
ENDIF ()
