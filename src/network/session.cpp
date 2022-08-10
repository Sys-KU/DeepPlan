#include <network/session.h>
#include <network/server_api.h>

#include <future>

namespace network {

SrvSession::SrvSession(boost::asio::io_service& io_service, MessageQueue& messages)
  : Session(io_service),
    messages_(messages) {};

message_rx* SrvSession::new_rx_message(uint64_t hdr_len, uint64_t body_len,
                                       uint64_t req_id, uint64_t msg_type) {
  message_rx* msg_rx;

  if (msg_type == REQ_INFERENCE) {
    auto msg = new msg_inference_req_rx();
    msg->set_req_id(req_id);
    msg->set_body_len(body_len);

    msg_rx = msg;
  }
  else if (msg_type == REQ_UPLOAD_MODEL) {
    auto msg = new msg_upload_model_req_rx();
    msg->set_req_id(req_id);

    msg_rx = msg;
  }
  else if(msg_type == REQ_CLOSE) {
    auto msg = new msg_close_req_rx();
    msg->set_req_id(req_id);

    msg_rx = msg;
  }

  return msg_rx;
}

bool SrvSession::completed_receive(message_connection* conn, message_rx* req) {
  bool is_continue = true;

  if (auto infer = dynamic_cast<msg_inference_req_rx*>(req)) {
    auto request = new serverapi::InferenceRequest();
    infer->get(*request);

    messages_.push({this, request});
  }
  else if (auto upload_model = dynamic_cast<msg_upload_model_req_rx*>(req)) {
    auto request = new serverapi::UploadModelRequest();
    upload_model->get(*request);

    messages_.push({this, request});
  }
  else if (auto close = dynamic_cast<msg_close_req_rx*>(req)) {
    auto response = new serverapi::CloseResponse();

    response->req_id = req->get_rx_req_id();
    response->dummy = 0;

    send_response(response);

    is_continue = false;
  }

  delete req;

  return is_continue;
}

void SrvSession::completed_transmit(message_connection* conn, message_tx* req) {
}

void SrvSession::send_response(serverapi::Response* response) {
  message_tx* msg_tx;

  if (auto infer = dynamic_cast<serverapi::InferenceResponse*>(response)) {
    auto infer_rsp = new msg_inference_rsp_tx();

    infer_rsp->set(*infer);
    msg_tx = infer_rsp;
  }
  else if (auto upload_model = dynamic_cast<serverapi::UploadModelResponse*>(response)) {
    auto upload_model_rsp = new msg_upload_model_rsp_tx();

    upload_model_rsp->set(*upload_model);
    msg_tx = upload_model_rsp;
  }
  else if (auto close = dynamic_cast<serverapi::CloseResponse*>(response)) {
    auto close_rsp = new msg_close_rsp_tx();

    close_rsp->set(*close);
    msg_tx = close_rsp;
  }

  msg_tx_.send_message(*msg_tx);
}

ClientSession::ClientSession(boost::asio::io_service& io_service)
  : Session(io_service),
    request_seed_id(0),
    received_rsp_cnt(0) {}

std::future<serverapi::Response*> ClientSession::send_request_async(serverapi::Request& request, std::function<void(serverapi::Response*)> onSuccess) {
  auto promise = std::make_shared<std::promise<serverapi::Response*>>();
  auto cb = [this, promise, onSuccess](serverapi::Response* response) {
    onSuccess(response);
    promise->set_value(response);
  };

  message_tx* msg_tx;

  uint64_t request_id = request_seed_id++;

  request.req_id = request_id;
  requests[request_id] = cb;

  if (auto infer = dynamic_cast<serverapi::InferenceRequest*>(&request)) {
    auto infer_req = new msg_inference_req_tx();

    infer_req->set(*infer);
    msg_tx = infer_req;
  }
  else if (auto upload_model = dynamic_cast<serverapi::UploadModelRequest*>(&request)) {
    auto upload_model_req = new msg_upload_model_req_tx();

    upload_model_req->set(*upload_model);
    msg_tx = upload_model_req;
  }
  else if (auto close = dynamic_cast<serverapi::CloseRequest*>(&request)) {
    auto close_req = new msg_close_req_tx();

    close_req->set(*close);
    msg_tx = close_req;
  }

  msg_tx_.send_message(*msg_tx);

  return promise->get_future();
}

serverapi::Response* ClientSession::send_request(serverapi::Request& request, std::function<void(serverapi::Response*)> onSuccess) {
  return send_request_async(request, onSuccess).get();
}

void ClientSession::await_completion() {
  while (request_seed_id > received_rsp_cnt) {
    usleep(100000);
  }

  return;
}

message_rx* ClientSession::new_rx_message(uint64_t hdr_len, uint64_t body_len,
                                          uint64_t req_id, uint64_t msg_type) {
  message_rx* msg_rx;

  if (msg_type == RSP_INFERENCE) {
    auto msg = new msg_inference_rsp_rx();
    msg->set_req_id(req_id);

    msg_rx = msg;
  }
  else if (msg_type == RSP_UPLOAD_MODEL) {
    auto msg = new msg_upload_model_rsp_rx();
    msg->set_req_id(req_id);

    msg_rx = msg;
  }
  else if (msg_type == RSP_CLOSE) {
    auto msg = new msg_close_rsp_rx();
    msg->set_req_id(req_id);

    msg_rx = msg;
  }

  return msg_rx;
}

bool ClientSession::completed_receive(message_connection* conn, message_rx* req) {
  uint64_t req_id = req->get_rx_req_id();
  serverapi::Response* response;
  bool is_continue = true;

  if (auto infer = dynamic_cast<msg_inference_rsp_rx*>(req)) {
    auto response_ = new serverapi::InferenceResponse();
    infer->get(*response_);
    response = response_;
  }
  else if (auto upload_model = dynamic_cast<msg_upload_model_rsp_rx*>(req)) {
    auto response_ = new serverapi::UploadModelResponse();
    upload_model->get(*response_);
    response = response_;
  }
  else if (auto close = dynamic_cast<msg_close_rsp_rx*>(req)) {
    auto response_ = new serverapi::CloseResponse();
    close->get(*response_);

    is_continue = false;
    response = response_;
  }

  requests[req_id](response);
  received_rsp_cnt++;

  delete req;

  return is_continue;
}

void ClientSession::completed_transmit(message_connection* conn, message_tx* req) {
}

}
