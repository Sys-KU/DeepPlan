#include <server.h>

#include <network/session.h>
#include <network/server_api.h>

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

void ClientSession::send_request(serverapi::Request& request, std::function<void(void)> onSuccess) {
  message_tx* msg_tx;

  uint64_t request_id = request_seed_id++;

  request.req_id = request_id;
  requests[request_id] = onSuccess;

  if (auto infer = dynamic_cast<serverapi::InferenceRequest*>(&request)) {
    auto infer_req = new msg_inference_req_tx();

    infer_req->set(*infer);
    msg_tx = infer_req;
  }
  else if (auto close = dynamic_cast<serverapi::CloseRequest*>(&request)) {
    auto close_req = new msg_close_req_tx();

    close_req->set(*close);
    msg_tx = close_req;
  }

  msg_tx_.send_message(*msg_tx);
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
  else if (msg_type == RSP_CLOSE) {
    auto msg = new msg_close_rsp_rx();
    msg->set_req_id(req_id);

    msg_rx = msg;
  }

  return msg_rx;
}

bool ClientSession::completed_receive(message_connection* conn, message_rx* req) {
  uint64_t req_id = req->get_rx_req_id();
  uint64_t msg_type = req->get_rx_msg_type();

  requests[req_id]();
  received_rsp_cnt++;

  delete req;

  if (msg_type == RSP_CLOSE)
    return false;
  else
    return true;
}

void ClientSession::completed_transmit(message_connection* conn, message_tx* req) {
}

}
