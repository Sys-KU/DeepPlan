#pragma once

namespace serverapi {

class Request {
 public:
  virtual ~Request() {};
  uint64_t req_id;
};

class Response {
 public:
  virtual ~Response() {};
  uint64_t req_id;
};

class InferenceRequest : public Request {
 public:
  uint32_t model_id;
  uint32_t batch_size;
  size_t input_size;
  void* input;
};

class InferenceResponse : public Response {
 public:
  int32_t dummy;
};

class CloseRequest : public Request {
 public:
  int32_t dummy;
};

class CloseResponse : public Response {
 public:
  int32_t dummy;
};

}
