#include <client/workload.h>
#include <client/genzipf.h>

Workload::Workload(int concurrency, int rate, int n_requests,
                   std::string dist_type, std::string addr, std::string port)
    : concurrency(concurrency),
      rate(rate),
      n_requests(n_requests),
      _traces(n_requests),
      addr(addr),
      port(port) {
        std::minstd_rand gen(0);
        std::exponential_distribution<double> edist(rate);

        if (dist_type == "uniform") {
          std::uniform_int_distribution<> udist(0, concurrency-1);

          for (auto& trace : _traces) {
            trace.first = edist(gen);
            trace.second = udist(gen);
          }
        }
        else if (dist_type == "zipfian") {
          rand_val(1);
          for (auto& trace : _traces) {
            trace.first = edist(gen);
            trace.second = zipf(1.5, concurrency) - 1;
          }
        }
        else {
          throw std::runtime_error("Not found the matching dist_type");
        }
      };

Workload::Workload(int concurrency, int rate, int n_requests,
                   float alpha, std::string addr, std::string port)
    : concurrency(concurrency),
      rate(rate),
      n_requests(n_requests),
      _traces(n_requests),
      addr(addr),
      port(port) {
        std::minstd_rand gen(0);
        std::exponential_distribution<double> edist(rate);

        if (alpha < 0) {
          throw std::runtime_error("The alpha value should be greater than 0");
        }

        rand_val(1);
        for (auto& trace : _traces) {
          trace.first = edist(gen);
          trace.second = zipf(alpha, concurrency) - 1;
        }
      };

Workload::Workload(std::vector<unsigned>& rates,
                   std::string addr, std::string port)
  : _traces(0),
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

void Workload::run(std::vector<std::vector<char>>& inputs) {
  client.connect(addr, port);

  double t1, t2;
  t1 = util::now();

  for (auto& trace : _traces) {
    double interval = trace.first;
    int model_id = trace.second;

    usleep(interval*1e6);

    uint64_t t_send = util::now();
    auto onSuccess = [this, model_id, t_send](serverapi::Response* rsp) {
      auto response = dynamic_cast<serverapi::InferenceResponse*>(rsp);
      uint64_t t_receive = util::now();
      uint64_t latency = (t_receive-t_send) / 1e6;

      this->res_results.emplace_back(model_id, latency, response->infer_time);
      if (response->is_cold) this->cold_start_cnt++;
    };

    client.infer_async(inputs[model_id], model_id, onSuccess);
  }

  client.shutdown();

  t2 = util::now();
  this->elapsed_time = (t2-t1) / 1e6;  // ms
}

WorkloadResult Workload::result(int slo) {
  WorkloadResult result;

  std::vector<double> latencies;
  for (auto& res_result : res_results) {
    latencies.push_back(res_result.latency);
  }

  std::sort(latencies.begin(), latencies.end());

  int index_50 = latencies.size() * 0.5 - 1;
  int index_99 = latencies.size() * 0.99 - 1;
  int goodput_cnt = 0;

  for (auto& latency : latencies)
    if (latency <= slo) goodput_cnt++;

  result.throughput = n_requests / (elapsed_time / 1e3);  // requests per sec
  result.latency_50 = latencies[index_50];
  result.latency_99 = latencies[index_99];
  result.cold_rate = (double)cold_start_cnt / n_requests * 100;
  result.goodput_rate = (double)goodput_cnt / n_requests * 100;

  return result;
}

void Workload::dump(std::string dump_file) {
  std::ofstream ofs;

  ofs.open(dump_file);
  if (ofs.is_open()) {
    std::cout << "Dump response results into '" << dump_file << "'\n";
    for (auto& res_result : res_results) {
      ofs << res_result.model_id << ", " << res_result.infer_time / 1e6 << "\n";
    }
  }

  std::cout << "Success Dump\n";
}

ModelLoader::ModelLoader(std::vector<std::string> model_names, int n_models,
                         EngineType engine_type, ReclaimPolicy r_policy,
                         int mp_size, std::string addr, std::string port)
  : model_names(model_names),
    n_models(n_models),
    engine_type(engine_type),
    r_policy(r_policy),
    mp_size(mp_size),
    addr(addr),
    port(port) {};

void ModelLoader::run() {
  client.connect(addr, port);

  util::InputGenerator input_generator;

  inputs.resize(n_models);

  int n_models_per_type = n_models / model_names.size();
  for (int i = 0; i < n_models; i++) {
    input_generator.generate_input(model_names[i/n_models_per_type], 1, &inputs[i]);
  }

  client.upload_model(model_names, n_models, engine_type, r_policy, mp_size);

  for (int i = 0; i < n_models; i++) {
    auto onSuccess = [this](serverapi::Response* rsp) {};

    client.infer_async(inputs[i], i, onSuccess);
  }

  client.shutdown();
}

