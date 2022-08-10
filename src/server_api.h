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
  size_t input_size;
  void* input;
};

struct InferenceResponse : public Response {
 public:
  bool is_cold;
  int32_t dummy;
};

struct UploadModelRequest : public Request {
 public:
  std::string model_name;
  uint32_t n_models;
  uint32_t engine_type;
  uint32_t mp_size;
};

struct UploadModelResponse : public Response {
 public:
  int32_t dummy;
};

struct CloseRequest : public Request {
 public:
  int32_t dummy;
};

struct CloseResponse : public Response {
 public:
  int32_t dummy;
};

}
