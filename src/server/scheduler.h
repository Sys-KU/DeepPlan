#include <util.h>
#include <server/util.h>
#include <options.h>
#include <network/session.h>
#include <server/model_manager.h>
#include <server/worker.h>
#include <deepplan/model.h>
#include <optional>
#include <set>
#include <mutex>


class CFR {
 public:
  bool put(int model_id, int vruntime) {
    if(exist(model_id)) {
      return false;
    }

    auto item = std::make_pair(vruntime, model_id);

    auto iter = items.begin();
    for (iter; iter != items.end(); iter++) {
      if (iter->first > item.first) {
        auto pos =items.insert(iter, item);
        index.emplace(model_id, pos);
        break;
      }
    }

    if (iter == items.end()) {
      auto pos = items.insert(iter, item);
      index.emplace(model_id, pos);
    }


    return true;
  }

  bool exist(int model_id) {
    return (index.count(model_id)>0);
  }

  std::pair<int, int> pop() {
    auto item = items.front();
    index.erase(item.second);
    items.pop_front();
    return item;
  }

  void erase(int model_id) {
    assert(exist(model_id));
    auto itr = index.find(model_id);

    index.erase(itr);
    items.erase(itr->second);
  }

  size_t size() {
    return index.size();
  }

 private:
  // pair = {vruntime, model_id}
  std::list<std::pair<int, int>> items;

  // key = model_id, valud = iterator of items
  std::unordered_map<int, typename std::list<std::pair<int, int>>::iterator> index;
};


// Reference https://gitlab.mpi-sws.org/cld/ml/clockwork/
class WorkerTracker {
 private:
  struct Work { int id; uint64_t exec_time; };
  std::deque<Work> outstandings;

  uint64_t total_outstanding_time = 0UL;
  uint64_t work_begin = 0UL;
  std::mutex exec_mutex;

 public:
  WorkerTracker() {};

  uint64_t available() {
    std::lock_guard<std::mutex> guard(exec_mutex);
    return std::max(work_begin + total_outstanding_time, util::now());
  }

  void update(int id, uint64_t end_time) {
    std::lock_guard<std::mutex> guard(exec_mutex);
    if (outstandings.front().id == id) {
      auto work = outstandings.front();
      total_outstanding_time -= work.exec_time;
      work_begin = end_time;
      outstandings.pop_front();
    }
    else {
      auto it = outstandings.begin();
      for (it; it != outstandings.end(); it++) {
        if (it->id == id) {
          total_outstanding_time -= it->exec_time;
          work_begin += it->exec_time;
          outstandings.erase(it);
        }
      }
    }
  }

  void add_work(int id, uint64_t exec_time) {
    std::lock_guard<std::mutex> guard(exec_mutex);
    if (outstandings.empty()) {
      work_begin = std::max(work_begin, util::now());
    }
    outstandings.push_back({id, exec_time});
    total_outstanding_time += exec_time;
  }
};


class MemoryTracker {
  struct Memory { int id; int64_t size; };
  std::deque<Memory> mem_requests;

  int64_t total_mem_req_size = 0UL;
  int64_t mem_begin = 0UL;
  std::mutex mem_mutex;

 public:
  MemoryTracker() {};

  uint64_t get_mem() {
    std::lock_guard guard(mem_mutex);
    return mem_begin + total_mem_req_size;
  }

  void update(int id, uint64_t end_mem) {
    std::lock_guard guard(mem_mutex);
    if (mem_requests.front().id == id) {
      auto request = mem_requests.front();
      total_mem_req_size -= request.size;
      mem_begin = end_mem;
      mem_requests.pop_front();
    }
    else {
      auto it = mem_requests.begin();
      for (it; it != mem_requests.end(); it++) {
        if (it->id == id) {
          total_mem_req_size -= it->size;
          mem_begin += it->size;
          mem_requests.erase(it);
        }
      }
    }
  }

  void load_mem(int id, uint64_t size) {
    std::lock_guard guard(mem_mutex);
    total_mem_req_size += size;
    mem_requests.push_back({id, static_cast<int64_t>(size)});
  }

  void reclaim_mem(int id, uint64_t size) {
    std::lock_guard guard(mem_mutex);
    total_mem_req_size -= size;
    mem_requests.push_back({id, static_cast<int64_t>(-size)});
  }

  void clear() {
    std::lock_guard guard(mem_mutex);
    mem_begin = 0UL;
  }
};


class Scheduler {
 public:
  Scheduler(int device, const ServerOptions& options,
            ModelPool* model_pool, std::string name="");

  void enqueue_request(
      serverapi::InferenceRequest* request,
      std::function<void(serverapi::InferenceResponse*)> cb,
      std::function<void(serverapi::TimeoutResponse*)> timeout_cb);

  void handle_requests();

  void handle_timeouts();

  deepplan::Model* find_model(int model_id, bool* is_cold);

  ReclaimingOutput preempt_models();

  void clear_models();

  void set_r_policy(ReclaimPolicy r_policy);

  void sync_setup();

  void stop();

  at::Device device;

  std::string name;

  ModelPool* model_pool;

 private:
  Worker* worker_;

  WorkerTracker exec;

  MemoryTracker mem;

  std::set<InferTask> requests_;

  std::queue<InferTask> timeouts_;

  const ServerOptions& options_;

  std::atomic_int action_seed_id = 0;

  // How far ahead, in nanoseconds, should the scheduler schedule.
  // Default is 10ms.
  uint64_t schedule_ahead = 1e7;

  size_t capacity_;

  uint64_t lag = 1e6;

  ReclaimPolicy r_policy_;
  util::LRUCache<int, deepplan::Model*>* running_models;
  RequestScoreboard* req_scoreboard = nullptr;
  std::vector<util::WindowBuf<int>> window_bufs;
  CFR* cfr;
  std::list<util::LRUCache<int, deepplan::Model*>*> partial_models_list;
};
