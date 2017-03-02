// Copyright 2015 The Gemmlowp Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// unpack_SSE.h: SSE-specialized unpacking the result blocks computed by 
// compute.h, storing them into the destination matrix.

#ifndef GEMMLOWP_INTERNAL_UNPACK_SSE_H_
#define GEMMLOWP_INTERNAL_UNPACK_SSE_H_

#include "allocator.h"
#include "block_params.h"
#include "output_avx2.h"
/* #include "pack_avx2.h" */

#include <cmath>

namespace gemmlowp {

// Non-optimized routine
template <std::uint32_t numerator, std::uint32_t denominator>
void AVX2RoundingMultiplyByConstantFraction(__m256i* x) {
  if (numerator == denominator) {
    return; 
  }
  std::int32_t x_array[8];
  _mm256_storeu_si256((__m256i*) &x_array[0], *x);
  for (int i=0; i<8; ++i) {
    x_array[i] = RoundingMultiplyByConstantFraction<numerator, denominator>(x_array[i]);
  }
  *x = _mm256_lddqu_si256(reinterpret_cast<__m256i*>(&x_array[0]));
}

template <typename tScalar, VectorShape tShape>
__m256i get_m256i_and_inc(
    ConstIterator<VectorMap<tScalar, tShape>>* iterator) {
  const __m256i result = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(iterator->get()));
  *iterator += 8;
  return result;
}

template <typename tScalar, VectorShape tShape>
__m256i get_m256i_and_inc(
    ConstIterator<VectorDup<tScalar, tShape>>* iterator) {
  const __m256i result = Dup<__m256i>(**iterator);
  // Increment really does nothing for VectorDup.
  *iterator += 8;
  return result;
}

template <typename BitDepthParams, typename PackedResultType, 
          typename OutputScalar, typename LhsOffset, typename RhsOffset,
          typename OutputPipelineType>
struct UnpackResultImpl<BitDepthParams,
                        MatrixMap<OutputScalar, MapOrder::ColMajor>,
                        PackedResultType, LhsOffset, RhsOffset,
                        OutputPipelineType> {
  typedef MatrixMap<OutputScalar, MapOrder::ColMajor> ResultBlockType;
  static void Unpack(ResultBlockType* dst, const MatrixBlockBounds& dst_block,
                     const PackedResultType& src, int depth,
                     const std::int32_t* lhs_sums_of_each_slice,
                     const std::int32_t* rhs_sums_of_each_slice,
                     const LhsOffset& lhs_offset, const RhsOffset& rhs_offset,
                     const OutputPipelineType& output_pipeline) {
    ScopedProfilingLabel label("optimized path (AVX2)");
    assert(dst_block.start_row >= 0);
    assert(dst_block.start_row + dst_block.rows <= dst->rows());
    assert(dst_block.start_col >= 0);
    assert(dst_block.start_col + dst_block.cols <= dst->cols());
    auto src_map = src.Map();
    // No top-level blocking in the depth dimension at the moment.
    // Too much loss of precision.
    const int kLhsBits = BitDepthParams::LhsBitDepth::kBits;
    const int kRhsBits = BitDepthParams::RhsBitDepth::kBits;
    const std::int32_t kLhsMax = (1 << kLhsBits) - 1;
    const std::int32_t kRhsMax = (1 << kRhsBits) - 1;
    __m256i depth_xmm = Dup<__m256i>((int32_t)depth);

    OutputPipelineExecutor<OutputPipelineType, AVX2FragmentInt32x8x1>
      int32x8x1_output_pipeline_executor(output_pipeline);
    OutputPipelineExecutor<OutputPipelineType, FragmentInt32x1x1>
      output_pipeline_executor(output_pipeline);

    for (int c = 0; c < dst_block.cols; c++) {
      int c_dst = c + dst_block.start_col;
      const std::int32_t* src_ptr = src_map.data(0, c);
      const std::int32_t* lhs_sums_of_each_slice_ptr = lhs_sums_of_each_slice;
      auto lhs_offset_iter = const_iterator(lhs_offset, dst_block.start_row);
      const std::int32_t rhs_offset_c = rhs_offset(c_dst);
      const std::int32_t rhs_sums_of_each_slice_c = rhs_sums_of_each_slice[c];

      __m256i rhs_offset_xmm = Dup<__m256i>(rhs_offset_c);
      __m256i rhs_sums_xmm   = Dup<__m256i>(rhs_sums_of_each_slice_c);

      // Handle 8 values at once for higher performance
      int dst_rows_aligned8 = RoundDown<8>(dst_block.rows);
      for (int r = 0; r < dst_rows_aligned8; 
           r += 8, src_ptr += 8, lhs_sums_of_each_slice_ptr += 8) {
        int r_dst = r + dst_block.start_row;

        // xx term: src_map(r:r+4,c)
        __m256i term_xx_xmm = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_ptr));
        AVX2RoundingMultiplyByConstantFraction<255 * 255, kLhsMax * kRhsMax> (&term_xx_xmm);
        // x1 term: lhs_sums_of_each_slice[r:r+4] * rhs_offset(c_dst)
        __m256i term_x1_xmm = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(lhs_sums_of_each_slice_ptr));
        term_x1_xmm = Mul(rhs_offset_xmm, term_x1_xmm);
        AVX2RoundingMultiplyByConstantFraction<255, kLhsMax> (&term_x1_xmm);
        // 1x term: rhs_sums_of_each_slice[c] * lhs_offset(r_dst:r_dst+3)
        __m256i lhs_offset_xmm = get_m256i_and_inc(&lhs_offset_iter);
        __m256i term_1x_xmm = Mul(rhs_sums_xmm, lhs_offset_xmm);
        AVX2RoundingMultiplyByConstantFraction<255, kRhsMax> (&term_1x_xmm);
        // 11 term: lhs_offset(r_dst:r_dst+3) * rhs_offset(c_dst) * depth
        __m256i term_11_xmm_ = Mul(rhs_offset_xmm, lhs_offset_xmm);
        __m256i term_11_xmm  = Mul(depth_xmm, term_11_xmm_);

        // sum xx, x1, 1x, 11
        __m256i sum_xmm = Add(
            Add(term_xx_xmm, term_x1_xmm), 
            Add(term_1x_xmm, term_11_xmm));
        AVX2FragmentInt32x8x1 f(sum_xmm);
        int32x8x1_output_pipeline_executor.Execute(f, dst, r_dst, c_dst);
      }

      for (int r = dst_rows_aligned8; r < dst_block.rows; r++) {
        int r_dst = r + dst_block.start_row;
        // To understand this code, read
        //   doc/low-precision.txt
        //   doc/less-than-8-bit.txt
        // We have 4 terms to sum: xx, x1, 1x, 11.
        // In case of requantization, we first need to scale them back
        // to the original scale, using RoundingMultiplyByConstantFraction.
        std::int32_t raw_xx = src_map(r, c);
        std::int32_t raw_x1 = lhs_sums_of_each_slice[r] * rhs_offset(c_dst);
        std::int32_t raw_1x = rhs_sums_of_each_slice[c] * lhs_offset(r_dst);
        std::int32_t term_xx =
            RoundingMultiplyByConstantFraction<255 * 255, kLhsMax * kRhsMax>(
                raw_xx);
        std::int32_t term_x1 =
            RoundingMultiplyByConstantFraction<255, kLhsMax>(raw_x1);
        std::int32_t term_1x =
            RoundingMultiplyByConstantFraction<255, kRhsMax>(raw_1x);
        std::int32_t term_11 = lhs_offset(r_dst) * rhs_offset(c_dst) * depth;
        // Sum the 4 terms.
        FragmentInt32x1x1 sum = term_xx + term_x1 + term_1x + term_11;
        output_pipeline_executor.Execute(sum, dst, r_dst, c_dst);
      }
    }
  }
};
}  // namespace gemmlowp


#endif  // GEMMLOWP_INTERNAL_UNPACK_H_
