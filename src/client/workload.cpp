#include <client/workload.h>

Workload::Workload(std::string model_name, int concurrency, int rate,
                   int n_requests, std::string addr, std::string port)
    : model_name(model_name),
      concurrency(concurrency),
      rate(rate),
      n_requests(n_requests),
      _traces(n_requests),
      addr(addr),
      port(port) {
        std::minstd_rand gen(0);
        std::uniform_int_distribution<> udist(0, concurrency-1);
        std::exponential_distribution<double> edist(rate);

        for (auto& trace : _traces) {
          trace.first = edist(gen);
          trace.second = udist(gen);
        }
      };

Workload::Workload(std::string model_name, std::vector<unsigned>& rates,
                   std::string addr, std::string port)
  : model_name(model_name),
    _traces(0),
    addr(addr),
    port(port) {
      std::minstd_rand gen(0);

      int cnt = 0;
      for (int i = 0; i < rates.size(); i++) {
        double itv = 0;
        std::exponential_distribution<double> edist(rates[i]/60.0);
        cnt += rates[i];

        itv = edist(gen);
        while (itv < 60) {
          _traces.push_back({itv, i});
          itv += edist(gen);
        }
      }

      sort(_traces.begin(), _traces.end(),
          [](auto& a, auto& b) { return a.first < b.first;});

      for (int i = _traces.size()-1; i > 0; i--) {
        _traces[i].first -= _traces[i-1].first;
      }

      n_requests = _traces.size();
    };

void Workload::run() {
  client.connect(addr, port);

  util::InputGenerator input_generator;

  std::vector<char> input;
  input_generator.generate_input(model_name, 1, &input);

  for (auto& trace : _traces) {
    double interval = trace.first;
    int model_id = trace.second;

    usleep(interval*1e6);

    uint64_t t_send = util::now();
    auto onSuccess = [this, t_send](serverapi::Response* rsp) {
      auto response = dynamic_cast<serverapi::InferenceResponse*>(rsp);
      uint64_t t_receive = util::now();
      uint64_t latency = (t_receive-t_send) / 1e6;

      this->latencies.push_back(latency);
      if (response->is_cold) this->cold_start_cnt++;
    };

    client.infer_async(input, model_id, onSuccess);
  }

  client.shutdown();
}

WorkloadResult Workload::result(int slo) {
  WorkloadResult result;

  std::sort(latencies.begin(), latencies.end());

  int index_99 = latencies.size() * 0.99 - 1;
  int goodput_cnt = 0;

  for (auto& latency : latencies)
    if (latency <= slo) goodput_cnt++;

  result.latency_99 = latencies[index_99];
  result.cold_rate = (double)cold_start_cnt / n_requests * 100;
  result.goodput_rate = (double)goodput_cnt / n_requests * 100;

  return result;
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

