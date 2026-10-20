// Text signatures shared by the loopbench Velox functions. The Python reference in
// loopbench/textsig.py implements the same arithmetic and must stay bit-identical.
//
//   words        maximal runs of ASCII [A-Za-z0-9], ASCII-lowercased
//   word hash    XXH64(word, seed 0)
//   n-gram hash  mix64(fold(h = h * P + word_hash)) over n consecutive words
//   MinHash      perm p: min over distinct shingles s of mix64(s ^ a_p),
//                a_p = mix64(seed ^ (p + 1) * GOLDEN); band b folds its rows with mix64
#pragma once

#include <algorithm>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#define XXH_INLINE_ALL
#include "xxhash.h"

namespace loopbench {

constexpr uint64_t kPrime = 1099511628211ULL;
constexpr uint64_t kGolden = 0x9E3779B97F4A7C15ULL;

inline uint64_t mix64(uint64_t z) {
  z ^= z >> 30;
  z *= 0xBF58476D1CE4E5B9ULL;
  z ^= z >> 27;
  z *= 0x94D049BB133111EBULL;
  z ^= z >> 31;
  return z;
}

inline void wordHashes(std::string_view text, std::vector<uint64_t>& out) {
  out.clear();
  std::string word;
  auto flush = [&] {
    if (!word.empty()) {
      out.push_back(XXH64(word.data(), word.size(), 0));
      word.clear();
    }
  };
  for (unsigned char c : text) {
    if (c >= 'A' && c <= 'Z') {
      word.push_back(static_cast<char>(c - 'A' + 'a'));
    } else if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')) {
      word.push_back(static_cast<char>(c));
    } else {
      flush();
    }
  }
  flush();
}

inline void ngramHashes(const std::vector<uint64_t>& words, int64_t n, std::vector<uint64_t>& out) {
  out.clear();
  if (n <= 0 || static_cast<int64_t>(words.size()) < n) {
    return;
  }
  for (size_t i = 0; i + n <= words.size(); ++i) {
    uint64_t h = 0;
    for (int64_t j = 0; j < n; ++j) {
      h = h * kPrime + words[i + j];
    }
    out.push_back(mix64(h));
  }
}

inline void minhashBands(
    std::vector<uint64_t>& shingles, int64_t bands, int64_t rows, uint64_t seed,
    std::vector<uint64_t>& out) {
  out.clear();
  if (shingles.empty()) {
    return;
  }
  std::sort(shingles.begin(), shingles.end());
  shingles.erase(std::unique(shingles.begin(), shingles.end()), shingles.end());
  std::vector<uint64_t> mins(bands * rows);
  for (int64_t p = 0; p < bands * rows; ++p) {
    const uint64_t a = mix64(seed ^ (static_cast<uint64_t>(p + 1) * kGolden));
    uint64_t m = UINT64_MAX;
    for (uint64_t s : shingles) {
      m = std::min(m, mix64(s ^ a));
    }
    mins[p] = m;
  }
  for (int64_t b = 0; b < bands; ++b) {
    uint64_t h = mix64(static_cast<uint64_t>(b + 1) * kGolden);
    for (int64_t r = 0; r < rows; ++r) {
      h = mix64(h ^ mins[b * rows + r]);
    }
    out.push_back(h);
  }
}

} // namespace loopbench
