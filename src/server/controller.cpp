#include <server/controller.h>
#include <server/worker.h>
#include <network/session.h>
#include <util.h>
#include <options.h>
#include <deepplan/engine.h>
#include <deepcache/engine.h>

#include <thread>

Controller::Controller(network::MessageQueue& messages, const ServerOptions& options)
  : messages_(messages),
    options_(options),
    alive(false) {init();};

void Controller::init() {
  deepcache::Init();
  deepplan::Init();

  alive = true;
  ctrl_thr = std::thread(std::bind(&Controller::run, this));

  int rank = torch::cuda::device_count();
  workers.resize(rank);
  for (int i = 0; i < workers.size(); i++) {
    workers[i] = new Worker(i, options_);
  }
}

void Controller::run() {
  while (alive) {
    network::Message message;

    if (messages_.try_pop(message)) {
      if (auto infer = dynamic_cast<serverapi::InferenceRequest*>(message.req)) {
        int model_id = infer->model_id;
        int n_workers = workers.size();
        int worker_id;

        worker_id = model_id % n_workers;
        infer->model_id = model_id / n_workers;

        auto cb = [message](serverapi::InferenceResponse* response) {
          message.srv_session->send_response(response);
        };

        workers[worker_id]->infer(infer, cb);
      }
      else if (auto upload_model = dynamic_cast<serverapi::UploadModelRequest*>(message.req)) {
        std::vector<std::string> model_names = upload_model->model_names;
        int n_models = upload_model->n_models;
        EngineType engine_type = static_cast<EngineType>(upload_model->engine_type);
        ReclaimPolicy r_policy = static_cast<ReclaimPolicy>(upload_model->r_policy);
        int mp_size = upload_model->mp_size;

        auto response = new serverapi::UploadModelResponse();

        setup_models(model_names, n_models, engine_type, r_policy, mp_size);

        response->req_id = upload_model->req_id;
        message.srv_session->send_response(response);
      }
    }
  }
}

void Controller::setup_models(std::vector<std::string> model_names, int n_models,
                              EngineType engine_type, ReclaimPolicy r_policy, int mp_size) {
  int n_workers = workers.size();
  bool should_setup = false;

  // Update if the setting parameters are different
  if ((model_names_ != model_names) ||
      (n_models_ != n_models) ||
      (engine_type_ != engine_type) ||
      (r_policy_ != r_policy) ||
      (mp_size_ != mp_size)) {
    should_setup = true;
  }

  if (should_setup) {
    std::vector<std::vector<int>> partitions(n_workers);

    for (int i = 0; i < n_workers; i++) {
      std::vector<int> p;
      for (int d = 0; d < mp_size; d++)
        p.push_back((i + 2*d) % n_workers);

      partitions[i] = p;
    }

    std::cout << "Setup total " << n_models << " new models with "
              << n_workers << " workers\n";
    if ((model_names_ == model_names) &&
        (engine_type_ == engine_type) &&
        (mp_size_ == mp_size)) {
      // If requested n_models is less than current n_models,
      // the models in Device memory are not freed,
      // but are just unloaded into the host memory.
      for (int i = 0; i < n_workers; i++) {
        workers[i]->clear_models();
        workers[i]->set_r_policy(r_policy);
      }
      if (n_models_ < n_models) {
        int n_models_per_worker = (n_models - n_models_) / n_workers;
        for (int i = 0; i < n_workers; i++) {
          workers[i]->add_models(model_names, n_models_per_worker,
                                 engine_type, partitions[i]);
        }
      }

      n_models = std::max(n_models, n_models_);
    }
    else {
      int n_models_per_worker = n_models / n_workers;
      for (int i = 0; i < n_workers; i++) {
        workers[i]->free_models();
        workers[i]->init_model_manager(engine_type);
        workers[i]->add_models(model_names, n_models_per_worker,
                               engine_type, partitions[i]);
        workers[i]->set_r_policy(r_policy);
      }
    }

    model_names_ = model_names;
    n_models_ = n_models;
    engine_type_ = engine_type;
    mp_size_ = mp_size;
    r_policy_ = r_policy;

    std::cout << "Modele setup complete\n";
  }

  return;
}

void Controller::shutdown() {
  alive = false;
  if (ctrl_thr.joinable())
    ctrl_thr.join();

  for (auto worker : workers)
    worker->stop();
}
