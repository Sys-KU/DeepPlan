#include <util.h>

namespace util {

std::uint64_t now() {
  return nanos(hrt());
}

time_point hrt()
{
  return std::chrono::steady_clock::now();
}

time_point epoch = hrt();

uint64_t epoch_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
    std::chrono::system_clock::now().time_since_epoch()).count();


std::uint64_t nanos(time_point t) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(t - epoch).count() + epoch_time;
}

size_t getModuleSize(ScriptModule module, bool ignore_cuda) {
  size_t size = 0;
  for (auto param : module.parameters()) {
    if (ignore_cuda && param.is_cuda()) continue;
    size += param.nbytes();
  }
  return size;
}

}
