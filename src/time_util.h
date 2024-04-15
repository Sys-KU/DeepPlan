#include <chrono>
#include <cstdint>

namespace util {

typedef std::chrono::steady_clock::time_point time_point;

time_point hrt();

std::uint64_t now();

std::uint64_t nanos(time_point t);

}  // namespace util
