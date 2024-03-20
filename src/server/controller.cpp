#include <server/controller.h>
#include <server/worker.h>
#include <network/session.h>
#include <util.h>
#include <time_util.h>
#include <options.h>
#include <deepplan/engine.h>
#include <sstream>

#include <thread>

Controller::Controller(network::MessageQueue& messages, const ServerOptions& options)
  : messages_(messages),
    options_(options),
    alive(false) {init();};

void Controller::init() {
  deepplan::Init();

  int rank = torch::cuda::device_count();

  auto model_repo = std::getenv("PLAN_REPO");
  if (model_repo == nullptr) {
    std::cerr << "PLAN_REPO variable not set, exiting\n";
    exit(EXIT_FAILURE);
  }

  model_manager = new ModelManager(model_repo, rank);
  schedulers.resize(rank);
  for (int i = 0; i < schedulers.size(); i++) {
    schedulers[i] = new Scheduler(i, options_, model_manager->model_pools[i]);
  }

  alive = true;
  ctrl_thr = std::thread(std::bind(&Controller::run, this));
}

void Controller::run() {
  while (alive) {
    network::Message message;

    while (messages_.try_pop(message)) {
      if (auto infer = dynamic_cast<serverapi::InferenceRequest*>(message.req)) {
        int model_id = infer->model_id;
        int n_workers = schedulers.size();
        int worker_id;

        worker_id = model_id % n_workers;
        infer->model_id = model_id / n_workers;

        auto cb = [message](serverapi::InferenceResponse* response) {
          message.srv_session->send_response(response);
        };

        auto timeout_cb = [message](serverapi::TimeoutResponse* response) {
          message.srv_session->send_response(response);
        };

        schedulers[worker_id]->enqueue_request(infer, cb, timeout_cb);
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

    for (auto scheduler : schedulers) {
      scheduler->handle_requests();
    }
    for (auto scheduler : schedulers) {
      scheduler->handle_timeouts();
    }

    usleep(10);
  }
}

void Controller::setup_models(std::vector<std::string> model_names,
                              int n_models,
                              EngineType engine_type,
                              ReclaimPolicy r_policy,
                              int mp_size) {
  for (int i = 0; i < schedulers.size(); i++) {
    schedulers[i]->clear_models();
    schedulers[i]->set_r_policy(r_policy);
  }

  bool updated = model_manager->setup(model_names, n_models, engine_type, mp_size);
  if (updated) {
    for (int i = 0; i < schedulers.size(); i++) {
      schedulers[i]->sync_setup();
    }
  }

  std::cout << "Model setup completed\n";

  return;
}

void Controller::shutdown() {
  alive = false;
  if (ctrl_thr.joinable())
    ctrl_thr.join();

  for (auto scheduler : schedulers) {
    scheduler->stop();
  }
}
