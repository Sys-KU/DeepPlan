#pragma once
#include <iostream>
#include <google/protobuf/text_format.h>

namespace network {
class message_tx {
 public:
  virtual uint64_t get_tx_hdr_len() const = 0;
  virtual uint64_t get_tx_body_len() const = 0;
  virtual uint64_t get_tx_req_id() const = 0;
  virtual uint64_t get_tx_msg_type() const = 0;
  virtual const void* tx_body_buf() = 0;

  virtual void serialize_header(void* dest) = 0;
};

class message_rx {
 public:
  virtual void header_received(const void* hdr, size_t hdr_len) = 0;
  virtual uint64_t get_rx_body_len() const = 0;
  virtual uint64_t get_rx_req_id() const = 0;
  virtual uint64_t get_rx_msg_type() const = 0;
  virtual void* rx_body_buf() = 0;

  virtual void body_buf_received(size_t len) = 0;
};

template <uint64_t TMsgType, class TMsg, class TReq>
class msg_protobuf_tx : public message_tx {
 protected:
  uint64_t req_id_;

 public:
  TMsg msg;
  static const uint64_t MsgType = TMsgType;

  void set_req_id(uint64_t req_id) { req_id_ = req_id; };

  virtual uint64_t get_tx_hdr_len() const { return msg.ByteSizeLong(); };
  virtual uint64_t get_tx_body_len() const { return 0; };
  virtual uint64_t get_tx_req_id() const { return req_id_; };
  virtual uint64_t get_tx_msg_type() const { return MsgType; };

  virtual void serialize_header(void* dest) {
    msg.SerializeToArray(dest, get_tx_hdr_len());
  }

  virtual const void* tx_body_buf() {
    throw "Should not be called";
  }

  virtual void set(TReq &request) = 0;
};

template <uint64_t TMsgType, class TMsg, class TRsp>
class msg_protobuf_rx : public message_rx {
 protected:
  uint64_t req_id_;

 public:
  TMsg msg;
  static const uint64_t MsgType = TMsgType;

  virtual void header_received(const void* hdr, size_t hdr_len) {
    if (!msg.ParseFromArray(hdr, hdr_len))
      std::cerr << "parsing failed\n";
  }

  void set_req_id(uint64_t req_id) { req_id_ = req_id; };

  virtual uint64_t get_rx_req_id() const { return req_id_; };
  virtual uint64_t get_rx_body_len() const { return 0; };
  virtual uint64_t get_rx_msg_type() const { return MsgType; };

  virtual void* rx_body_buf() {
    throw "Should not be called";
  }

  virtual void body_buf_received(size_t len) {
    throw "Should not be called";
  }

  virtual void get(TRsp& response) = 0;

};

template <uint64_t TMsgType, class TMsg, class TRsp>
class msg_protobuf_tx_with_body : public msg_protobuf_tx<TMsgType, TMsg, TRsp> {
 protected:
  size_t body_len_ = 0;
  void* body_ = nullptr;

 public:
  virtual void set_body_len(size_t body_len) { body_len_ = body_len; }

  virtual uint64_t get_tx_body_len() const { return body_len_; }

  virtual const void* tx_body_buf() {
    return body_;
  }

};

template <uint64_t TMsgType, class TMsg, class TRsp>
class msg_protobuf_rx_with_body : public msg_protobuf_rx<TMsgType, TMsg, TRsp> {
 protected:
  size_t body_len_ = 0;
  void* body_ = nullptr;

 public:
  virtual void set_body_len(size_t body_len) {
    body_len_ = body_len;
    body_ = new uint8_t[body_len];
  }

  virtual uint64_t get_rx_body_len() const { return body_len_; }

  virtual void* rx_body_buf() {
    return body_;
  }

  virtual void body_buf_received(size_t len) {
  }
};

}
