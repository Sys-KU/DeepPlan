#pragma once

namespace serverapi {

struct Request {
 public:
  virtual ~Request() {};
  uint64_t req_id;
};

struct Response {
 public:
  virtual ~Response() {};
  uint64_t req_id;
};

struct InferenceRequest : public Request {
 public:
  uint32_t model_id;
  uint32_t batch_size;
  uint64_t deadline;
  uint64_t arrival_time;
  size_t input_size;
  bool disable_timeout;
  void* input;
};

struct InferenceResponse : public Response {
 public:
  bool is_cold;
  double infer_time;
  uint64_t arrival_time;
  uint64_t response_time;
  uint64_t deadline;
};

struct UploadModelRequest : public Request {
 public:
  std::vector<std::string> model_names;
  uint32_t n_models;
  uint32_t engine_type;
  uint32_t r_policy;
  uint32_t mp_size;
};

struct UploadModelResponse : public Response {
};

struct CloseRequest : public Request {
};

struct CloseResponse : public Response {
};

struct TimeoutResponse : public Response {
};

}
