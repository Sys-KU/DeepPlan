#pragma once

namespace azure {

std::string get_trace_dir() {
  auto trace_dir = std::getenv("AZURE_TRACE_DIR");

  if (trace_dir == nullptr) { return ""; }
  return trace_dir == nullptr ? "" : std::string(trace_dir);
}

std::string get_trace_file(std::string trace_dir, int id) {
  std::stringstream ss;

  ss << trace_dir << "/invocations_per_function_md.anon.d";
  if (id < 10)
    ss << "0";
  ss << id << ".csv";

  return ss.str();
}

std::string get_trace(int id) {
  std::string trace_dir = get_trace_dir();

  if (trace_dir == "") {
    std::cerr << "AZURE_TRACE_DIR variable not set, exiting\n";
    exit(EXIT_FAILURE);
  }

  if (1 > id || id > 14) {
    std::cerr << "Azure workload_id must be between 1 and 14 inclusive. Got "
              << id << "\n";
    exit(EXIT_FAILURE);
  }

  std::string trace_file = get_trace_file(trace_dir, id);

  return trace_file;
}

std::vector<std::string> split(std::string line) {
  std::vector<std::string> result;
  std::stringstream s(line);
  while (s.good()) {
    std::string substr;
    std::getline(s, substr, ',');
    result.push_back(substr);
  }
  return result;
}

std::vector<unsigned> process_trace_line(std::string line, unsigned start_index) {
  std::vector<std::string> splits = split(line);
  std::vector<unsigned> result;
  for (unsigned i = start_index; i < splits.size(); i++) {
    result.push_back(std::stoul(splits[i].c_str(), NULL, 10));
  }
  return result;
}

std::vector<std::vector<unsigned>> read_trace_data(std::string filename) {
  std::ifstream f(filename);

  std::vector<std::vector<unsigned>> results;
  std::vector<std::pair<int, int>> sizes;

  std::string line;
  std::getline(f, line); // Skip headers
  while (std::getline(f, line)) {
    auto traceline = process_trace_line(line, 4);
    int size = std::accumulate(traceline.begin(), traceline.end(), 0);
    sizes.push_back(std::make_pair(size, results.size()));
    results.push_back(traceline);
  }

  std::sort(sizes.begin(), sizes.end());

  std::vector<std::vector<unsigned>> ordered;
  for (int i = sizes.size()-1; i >= 0; i--) {
    ordered.push_back(results[sizes[i].second]);
  }

  return ordered;
}

std::vector<std::vector<unsigned>> load_trace(int workload_id = 1) {
  return read_trace_data(get_trace(workload_id));
}

std::vector<std::vector<unsigned>> scale_trace_rate(std::vector<std::vector<unsigned>>& traces, int rate) {
  std::vector<std::vector<unsigned>> scaled_traces(traces.size());

  unsigned total_size = 0;
  double total_rate = 0;
  double scale_ratio = 0;

  for (auto& trace : traces)
    total_size += std::accumulate(trace.begin(), trace.end(), 0);

  total_rate = total_size / 24.0 / 60.0 / 60.0; // caculate rate(r/s);
  scale_ratio = rate / total_rate;

  std::transform(traces.begin(), traces.end(), scaled_traces.begin(),
                [scale_ratio](auto vec) {
                  for (auto& v : vec) v *= scale_ratio;
                  return vec;
                });

  return scaled_traces;
}

std::vector<std::vector<unsigned>> scale_trace_size(std::vector<std::vector<unsigned>>& traces, int size) {
  std::vector<std::vector<unsigned>> scaled_traces(size);

  for (int i = 0; i < size; i++) {
    scaled_traces[i] = traces[i];
  }

  // Compress the traces size
  for (int i = size; i < traces.size(); i++) {
    for (int j = 0; j < traces[i].size(); j++) {
      scaled_traces[i % size][j] += traces[i][j];
    }
  }

  return scaled_traces;
}

std::vector<std::vector<unsigned>> load_scaled_trace(int rate, int size, int workload_id = 1) {
  auto traces = load_trace(workload_id);

  auto scaled_traces = scale_trace_rate(traces, rate);
  scaled_traces = scale_trace_size(scaled_traces, size);

  return scaled_traces;
}

template <typename T>
void transpose(std::vector<std::vector<T>> &m) {
  if (m.size() == 0)
    return;

  std::vector<std::vector<T>> trans_vec(m[0].size(), std::vector<T>());

  for (int i = 0; i < m.size(); i++) {
    for (int j = 0; j < m[i].size(); j++) {
      trans_vec[j].push_back(m[i][j]);
    }
  }

  m = trans_vec;
}

}
