#pragma once

#include <util.h>

class RequestScoreboard {
 public:
  RequestScoreboard(int num_models, int window_size)
    : scores_(num_models, 0),
      window_buf_(window_size) {}

  RequestScoreboard(int window_size)
    : window_buf_(window_size) {}

  void update_window(int model_id) {
    scores_[model_id]++;

    if (window_buf_.full()) {
      scores_[window_buf_.front()]--;
    }

    window_buf_.update(model_id);
  }

  int get_score(int model_id) {
    return scores_[model_id];
  }

  void clear() {
    window_buf_.clear();
    std::fill(scores_.begin(), scores_.end(), 0);
  }

  void resize(int num_models) {
    scores_.resize(num_models);
    std::fill(scores_.begin(), scores_.end(), 0);
  }


 private:
  util::WindowBuf<int> window_buf_;

  std::vector<int> scores_;
};
