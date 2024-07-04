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

#define SCHEDULE_AHEAD_DEFAULT 1e7
#define SCHEDULE_AHEAD_SYNC 1e6
#define LOAD_AHEAD_DEFAULT 3e7
#define LOAD_AHEAD_SYNC 1e6

struct Completion {
  Completion() {};
  Completion(int action_id, uint64_t end_time)
    : action_id(action_id), end_time(end_time) {};

  virtual ~Completion() {};

  int action_id;
  uint64_t end_time;
};


struct InferCompletion : public Completion {
  InferCompletion() {};
  InferCompletion(int action_id, uint64_t end_time, int model_id)
    : Completion(action_id, end_time), model_id(model_id) {};

  int model_id;
};

struct LoadCompletion : public Completion {
  LoadCompletion() {};
  LoadCompletion(
    int action_id,
    uint64_t end_time,
    uint64_t end_size)
    : Completion(action_id, end_time), end_size(end_size) {};

  uint64_t end_size;
};

struct ReclaimCompletion : public Completion {
  ReclaimCompletion() {};
  ReclaimCompletion(
    int action_id,
    uint64_t end_time,
    uint64_t end_size)
    : Completion(action_id, end_time), end_size(end_size) {};

  uint64_t end_size;
};


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

 public:
  WorkerTracker() {};

  uint64_t available(const uint64_t now) {
    if ((outstandings.size() > 0) &&
        (work_begin + outstandings.front().exec_time < now)) {
      // Outstanding work has mysteriously not completed
      work_begin = now - outstandings.front().exec_time;
    }
    return std::max(work_begin + total_outstanding_time, now);
  }

  void update(int id, uint64_t end_time) {
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
    if (outstandings.empty()) {
      work_begin = std::max(work_begin, util::now());
    }
    outstandings.push_back({id, exec_time});
    total_outstanding_time += exec_time;
  }
};


class MemoryTracker {
  struct Memory { int id; int model_id; uint64_t load_time; int64_t size;};
  std::deque<Memory> mem_requests;

  uint64_t total_loading_time = 0UL;
  int64_t total_mem_size = 0UL;
  uint64_t load_begin = 0UL;
  int64_t mem_begin = 0UL;

 public:
  MemoryTracker() {};

  uint64_t available(const uint64_t now) {
    if ((mem_requests.size() > 0) &&
        (load_begin + mem_requests.front().load_time < now)) {
      // Outstanding work has mysteriously not completed
      load_begin = now - mem_requests.front().load_time;
    }
    return std::max(load_begin + total_loading_time, now);
  }

  uint64_t get_mem() {
    return mem_begin + total_mem_size;
  }

  uint64_t end_time(int model_id) {
    auto it = mem_requests.begin();
    uint64_t end_time = load_begin;
    bool found = false;
    for (it; it != mem_requests.end(); it++) {
      end_time += it->load_time;
      if (it->model_id == model_id) {
        found = true;
        break;
      }
    }

    if (!found) {
      end_time = 0;
    }

    return end_time;
  }

  void update(int id, uint64_t end_time, uint64_t end_mem) {
    if (mem_requests.front().id == id) {
      auto request = mem_requests.front();
      total_loading_time -= request.load_time;
      total_mem_size -= request.size;
      load_begin = end_time;
      mem_begin = end_mem;
      mem_requests.pop_front();
    }
    else {
      auto it = mem_requests.begin();
      for (it; it != mem_requests.end(); it++) {
        if (it->id == id) {
          total_loading_time -= it->load_time;
          total_mem_size -= it->size;
          load_begin += it->load_time;
          mem_begin += it->size;
          mem_requests.erase(it);
          break;
        }
      }
    }
  }

  void load_mem(int id, int model_id, uint64_t load_time, uint64_t size) {
    if (mem_requests.empty()) {
      load_begin = std::max(load_begin, util::now());
    }
    mem_requests.push_back({id, model_id, load_time, static_cast<int64_t>(size)});
    total_loading_time += load_time;
    total_mem_size += size;
  }

  void reclaim_mem(int id, int model_id, uint64_t size) {
    total_mem_size -= size;
    mem_requests.push_back({id, model_id, 0UL, static_cast<int64_t>(-size)});
  }

  void clear() {
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

  void handle_load(const uint64_t now);

  void handle_exec(const uint64_t now);

  void handle_timeouts();

  void reclaim_gpu_memory(const size_t required_memory);

  deepplan::Model* find_model(int model_id);

  ReclaimingOutput preempt_model();

  void clear_models();

  void set_r_policy(ReclaimPolicy r_policy);

  void sync_setup();

  void stop();

  at::Device device;

  std::string name;

  ModelPool* model_pool;

  // How far ahead, in nanoseconds, should the scheduler schedule or load.
  uint64_t schedule_ahead;
  uint64_t load_ahead;
  bool allow_prefetch = true;

 private:
  Worker* worker_;

  WorkerTracker exec;

  MemoryTracker mem;

  std::multiset<InferTask> requests_;

  std::queue<InferTask> timeouts_;

  std::mutex tracker_mutex;

  std::queue<std::shared_ptr<Completion>> completion_queue_;

  const ServerOptions& options_;

  std::atomic_int action_seed_id = 0;

  size_t capacity_;

  uint64_t lag = 1e6;

  ReclaimPolicy r_policy_;
  util::LRUCache<int, deepplan::Model*>* running_models;
  std::vector<int> model_ref_cnts;
  RequestScoreboard* req_scoreboard = nullptr;
  std::vector<util::WindowBuf<int>> window_bufs;
  CFR* cfr;
  std::list<util::LRUCache<int, deepplan::Model*>*> partial_models_list;
};
