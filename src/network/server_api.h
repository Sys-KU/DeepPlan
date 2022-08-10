#pragma once

#include <deepcache.pb.h>
#include <server_api.h>
#include <network/message.h>

namespace network {

class msg_inference_req_tx : public msg_protobuf_tx_with_body<REQ_INFERENCE, InferenceReqProto, serverapi::InferenceRequest> {
 public:
  virtual void set(serverapi::InferenceRequest& request);
};

class msg_inference_req_rx : public msg_protobuf_rx_with_body<REQ_INFERENCE, InferenceReqProto, serverapi::InferenceRequest> {
 public:
  virtual void get(serverapi::InferenceRequest& request);
};

class msg_inference_rsp_tx : public msg_protobuf_tx<RSP_INFERENCE, InferenceRspProto, serverapi::InferenceResponse> {
 public:
  virtual void set(serverapi::InferenceResponse& response);
};

class msg_inference_rsp_rx : public msg_protobuf_rx<RSP_INFERENCE, InferenceRspProto, serverapi::InferenceResponse> {
 public:
  virtual void get(serverapi::InferenceResponse& response);
};

class msg_upload_model_req_tx : public msg_protobuf_tx<REQ_UPLOAD_MODEL, UploadModelReqProto, serverapi::UploadModelRequest> {
 public:
  virtual void set(serverapi::UploadModelRequest& request);
};

class msg_upload_model_req_rx : public msg_protobuf_rx<REQ_UPLOAD_MODEL, UploadModelReqProto, serverapi::UploadModelRequest> {
 public:
  virtual void get(serverapi::UploadModelRequest& request);
};

class msg_upload_model_rsp_tx : public msg_protobuf_tx<RSP_UPLOAD_MODEL, UploadModelRspProto, serverapi::UploadModelResponse> {
 public:
  virtual void set(serverapi::UploadModelResponse& response);
};

class msg_upload_model_rsp_rx : public msg_protobuf_rx<RSP_UPLOAD_MODEL, UploadModelRspProto, serverapi::UploadModelResponse> {
 public:
  virtual void get(serverapi::UploadModelResponse& response);
};

class msg_close_req_tx : public msg_protobuf_tx<REQ_CLOSE, CloseReqProto, serverapi::CloseRequest> {
 public:
  virtual void set(serverapi::CloseRequest& request);
};

class msg_close_req_rx : public msg_protobuf_rx<REQ_CLOSE, CloseReqProto, serverapi::CloseRequest> {
 public:
  virtual void get(serverapi::CloseRequest& request);
};

class msg_close_rsp_tx : public msg_protobuf_tx<RSP_CLOSE, CloseRspProto, serverapi::CloseResponse> {
 public:
  virtual void set(serverapi::CloseResponse& response);
};

class msg_close_rsp_rx : public msg_protobuf_rx<RSP_CLOSE, CloseRspProto, serverapi::CloseResponse> {
 public:
  virtual void get(serverapi::CloseResponse& response);
};

}

