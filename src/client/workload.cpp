#include <client/workload.h>

Workload::Workload(std::string model_name, int concurrency, int rate,
                   int n_requests, int mp_size, EngineType engine_type,
                   std::string addr, std::string port)
    : model_name(model_name),
      concurrency(concurrency),
      rate(rate),
      n_requests(n_requests),
      mp_size(mp_size),
      engine_type(engine_type),
      addr(addr),
      port(port) {};

void Workload::run() {
  client.connect(addr, port);

  // Set seed to 0;
  std::minstd_rand gen(0);
  std::uniform_int_distribution<> udist(0, concurrency-1);
  std::exponential_distribution<double> edist(rate);

  std::vector<int> model_ids(n_requests);
  std::vector<double> intervals(n_requests);

  for (auto& id : model_ids) id = udist(gen);
  for (auto& itv : intervals) itv = edist(gen);

  util::InputGenerator input_generator;

  std::vector<char> input;
  input_generator.generate_input(model_name, 1, &input);


  for (int i = 0; i < n_requests; i++) {
    int model_id = model_ids[i];
    double interval = intervals[i];

    usleep(interval*1e6);

    uint64_t t_send = util::now();
    auto onSuccess = [this, t_send](serverapi::Response* rsp) {
      uint64_t t_receive = util::now();
      uint64_t latency = (t_receive-t_send) / 1e6;
      this->latencies.push_back(latency);
    };

    client.infer_async(input, model_id, onSuccess);
  }

  client.shutdown();
}

std::vector<double> Workload::result() {
  std::sort(latencies.begin(), latencies.end());
  return latencies;
}

ModelLoader::ModelLoader(std::string model_name, int n_models, EngineType engine_type,
                         int mp_size, std::string addr, std::string port)
  : model_name(model_name),
    n_models(n_models),
    engine_type(engine_type),
    mp_size(mp_size),
    addr(addr),
    port(port) {};

void ModelLoader::run() {
  client.connect(addr, port);

  util::InputGenerator input_generator;

  std::vector<char> input;
  input_generator.generate_input(model_name, 1, &input);

  client.upload_model(model_name, n_models, engine_type, mp_size);

  for (int i = 0; i < n_models; i++) {
    auto onSuccess = [this](serverapi::Response* rsp) {};

    client.infer_async(input, i, onSuccess);
  }

  client.shutdown();
}

