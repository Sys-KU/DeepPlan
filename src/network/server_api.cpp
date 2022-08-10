#include <network/server_api.h>

namespace network {

void msg_inference_req_tx::set(serverapi::InferenceRequest& request) {
  set_req_id(request.req_id);
  msg.set_req_id(request.req_id);
  msg.set_model_id(request.model_id);
  msg.set_batch_size(request.batch_size);
  body_len_ = request.input_size;
  body_ = request.input;
}

void msg_inference_req_rx::get(serverapi::InferenceRequest& request) {
  request.req_id = get_rx_req_id();
  request.model_id = msg.model_id();
  request.batch_size = msg.batch_size();
  request.input_size = body_len_;
  request.input = body_;
}

void msg_inference_rsp_tx::set(serverapi::InferenceResponse& response) {
  set_req_id(response.req_id);
  msg.set_req_id(response.req_id);
  msg.set_is_cold(response.is_cold);
  msg.set_dummy(response.dummy);
}

void msg_inference_rsp_rx::get(serverapi::InferenceResponse& response) {
  response.req_id = get_rx_req_id();
  response.is_cold = msg.is_cold();
  response.dummy = msg.dummy();
}

void msg_upload_model_req_tx::set(serverapi::UploadModelRequest& request) {
  set_req_id(request.req_id);
  msg.set_req_id(request.req_id);
  msg.set_model_name(request.model_name);
  msg.set_n_models(request.n_models);
  msg.set_engine_type(request.engine_type);
  msg.set_mp_size(request.mp_size);
}

void msg_upload_model_req_rx::get(serverapi::UploadModelRequest& request) {
  request.req_id = get_rx_req_id();
  request.model_name = msg.model_name();
  request.n_models = msg.n_models();
  request.engine_type = msg.engine_type();
  request.mp_size = msg.mp_size();
}

void msg_upload_model_rsp_tx::set(serverapi::UploadModelResponse& response) {
  set_req_id(response.req_id);
  msg.set_req_id(response.req_id);
  msg.set_dummy(response.dummy);
}

void msg_upload_model_rsp_rx::get(serverapi::UploadModelResponse& response) {
  response.req_id = get_rx_req_id();
  response.dummy = msg.dummy();
}

void msg_close_req_tx::set(serverapi::CloseRequest& request) {
  set_req_id(request.req_id);
  msg.set_req_id(request.req_id);
  msg.set_dummy(request.dummy);
}

void msg_close_req_rx::get(serverapi::CloseRequest& request) {
  request.req_id = get_rx_req_id();
  request.dummy = msg.dummy();
}

void msg_close_rsp_tx::set(serverapi::CloseResponse& response) {
  set_req_id(response.req_id);
  msg.set_req_id(response.req_id);
  msg.set_dummy(response.dummy);
}

void msg_close_rsp_rx::get(serverapi::CloseResponse& response) {
  response.req_id = get_rx_req_id();
  response.dummy = msg.dummy();
}

}
