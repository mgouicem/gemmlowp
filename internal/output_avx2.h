// Copyright 2015 Google Inc. All Rights Reserved.
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

// output_avx2.h: optimized AVX2 specializations of the templates in output.h.

#ifndef GEMMLOWP_INTERNAL_OUTPUT_AVX2_H_
#define GEMMLOWP_INTERNAL_OUTPUT_AVX2_H_

#include <smmintrin.h>
#include "output.h"

#include <iostream>

namespace gemmlowp {

  typedef struct _int32x16x1_t {
    __m256i val[2];
  } int32x16x1_t;

  typedef struct _int32x32x1_t {
    __m256i val[4];
  } int32x32x1_t;
  
// Definitions of Fragment types wrapping AVX2 vector types.
typedef Fragment<__m256i, 8, 1, MapOrder::ColMajor> AVX2FragmentInt32x8x1;
typedef Fragment<int32x16x1_t, 16, 1, MapOrder::ColMajor> AVX2FragmentInt32x16x1;
typedef Fragment<int32x32x1_t, 32, 1, MapOrder::ColMajor> AVX2FragmentInt32x32x1;

typedef Fragment<uint64_t, 8, 1, MapOrder::ColMajor> AVX2FragmentUint8x8x1;
typedef Fragment<__m256i, 32, 1, MapOrder::ColMajor> AVX2FragmentUint8x32x1;

template <typename OutputStageType>
struct OutputStageEvalImpl<OutputStageType, AVX2FragmentInt32x32x1> {
  typedef AVX2FragmentInt32x32x1 InputType;
  typedef AVX2FragmentInt32x32x1 OutputType;
  typedef OutputStageEvalImpl<OutputStageType, AVX2FragmentInt32x8x1>
      ImplInt32x8;
  OutputStageEvalImpl(const OutputStageType& s) : impl_int32x8(s) {}

  OutputType Eval(InputType input, int row, int col) const {
    OutputType output;

    for (int i = 0; i < 4; i++) {
      output.data.val[i] =
          impl_int32x8.Eval(input.data.val[i], row + 4 * i, col);
    }
    return output;
  }

  ImplInt32x8 impl_int32x8;
};

// Implementation of OutputStageQuantizeDownInt32ToUint8Scale for
// AVX2FragmentInt32x8x1
template <>
struct OutputStageEvalImpl<OutputStageQuantizeDownInt32ToUint8Scale,
                           AVX2FragmentInt32x8x1> {
  typedef AVX2FragmentInt32x8x1 InputType;
  typedef AVX2FragmentInt32x8x1 OutputType;
  typedef OutputStageQuantizeDownInt32ToUint8Scale OutputStage;

  OutputStageEvalImpl(const OutputStage& s) : output_stage(s) {}

  OutputType Eval(InputType input, int, int) const {
    const std::int32_t result_shift = output_stage.result_shift;
    const __m256i result_mult_int = Dup<__m256i>(output_stage.result_mult_int);
    const __m256i result_offset = Dup<__m256i>(output_stage.result_offset);
    const __m256i a = Mul(
			  Add(input.data, result_offset),
			  result_mult_int);
    return RoundingDivideByPOT(a, result_shift);
  }

  const OutputStage& output_stage;
};

// Implementation of OutputStageQuantizeDownInt32ToUint8ScalePC for
// AVX2FragmentInt32x8x1
template <>
struct OutputStageEvalImpl<
    OutputStageQuantizeDownInt32ToUint8ScalePC<VectorShape::Col>,
    AVX2FragmentInt32x8x1> {
  typedef AVX2FragmentInt32x8x1 InputType;
  typedef AVX2FragmentInt32x8x1 OutputType;
  typedef OutputStageQuantizeDownInt32ToUint8ScalePC<VectorShape::Col>
      OutputStage;

  OutputStageEvalImpl(const OutputStage& s) : output_stage(s) {}

  OutputType Eval(InputType input, int row, int col) const {
    const std::int32_t result_shift = output_stage.result_shift;
    const __m256i result_mult_int =
      _mm256_lddqu_si256(reinterpret_cast<const __m256i*>(output_stage.result_mult_int.data(row)));
    const __m256i result_offset =
      _mm256_lddqu_si256(reinterpret_cast<const __m256i*>(output_stage.result_offset.data(row)));
    const __m256i a = Mul(
			  Add(input.data, result_offset),
			  result_mult_int);
    return RoundingDivideByPOT(a, result_shift);
  }

  const OutputStage& output_stage;
};

// Implementation of OutputStageQuantizeDownInt32ToUint8ScalePC for
// AVX2FragmentInt32x8x1
template <>
struct OutputStageEvalImpl<
    OutputStageQuantizeDownInt32ToUint8ScalePC<VectorShape::Row>,
    AVX2FragmentInt32x8x1> {
  typedef AVX2FragmentInt32x8x1 InputType;
  typedef AVX2FragmentInt32x8x1 OutputType;
  typedef OutputStageQuantizeDownInt32ToUint8ScalePC<VectorShape::Row>
      OutputStage;

  OutputStageEvalImpl(const OutputStage& s) : output_stage(s) {}

  OutputType Eval(InputType input, int row, int col) const {
    const std::int32_t result_shift = output_stage.result_shift;
    const __m256i result_mult_int =
      _mm256_lddqu_si256(reinterpret_cast<const __m256i*>(output_stage.result_mult_int.data(col)));
    const __m256i result_offset =
      _mm256_lddqu_si256(reinterpret_cast<const __m256i*>(output_stage.result_offset.data(row)));
    const __m256i a = Mul(
			  Add(input.data, result_offset),
			  result_mult_int);
    return RoundingDivideByPOT(a, result_shift);
  }

  const OutputStage& output_stage;
};

// Implementation of OutputStageQuantizeDownInt32ToUint8ScaleByFixedPoint for
// AVX2FragmentInt32x8x1
template <>
struct OutputStageEvalImpl<OutputStageQuantizeDownInt32ToUint8ScaleByFixedPoint,
                           AVX2FragmentInt32x8x1> {
  typedef AVX2FragmentInt32x8x1 InputType;
  typedef AVX2FragmentInt32x8x1 OutputType;
  typedef OutputStageQuantizeDownInt32ToUint8ScaleByFixedPoint OutputStage;

  OutputStageEvalImpl(const OutputStage& s) : output_stage(s) {}

  OutputType Eval(InputType input, int, int) const {
    const __m256i mulhigh_val = SaturatingRoundingDoublingHighMul(
        input.data,
	Dup<__m256i>(output_stage.result_fixedpoint_multiplier));
    const std::int32_t result_shift = output_stage.result_shift;
    const __m256i shifted_val = RoundingDivideByPOT(mulhigh_val,
						    result_shift);
    return Add(shifted_val, Dup<__m256i>(output_stage.result_offset_after_shift));
  }
  
  const OutputStage& output_stage;
};


// Implementation of OutputStageSaturatingCastToUint8 for AVX2FragmentInt32x8x1
template <>
struct OutputStageEvalImpl<OutputStageSaturatingCastToUint8,
                           AVX2FragmentInt32x8x1> {
  typedef AVX2FragmentInt32x8x1 InputType;
  typedef AVX2FragmentUint8x8x1 OutputType;
  typedef OutputStageSaturatingCastToUint8 OutputStage;

  OutputStageEvalImpl(const OutputStage&) {}

  OutputType Eval(InputType input, int, int) const {
    const __m256i zero = Dup<__m256i>(0);
    __m256i res_16 = _mm256_packs_epi32(input,
					_mm256_castsi128_si256(_mm256_extractf128_si256(input, 1)));
    //    __m256i unpacked_res_16 = _mm256_unpacklo_epi64(res_16, res_16);
    __m256i res_8  = _mm256_packus_epi16(res_16, zero);
    return _mm_cvtsi128_si64(_mm256_castsi256_si128(res_8));
  }
};

// In the case of OutputStageSaturatingCastToUint8, the handling of
// AVX2FragmentInt32x16x1 data can be made much more efficient by handling
// it all at once, instead of as 4 separate int32x8 values as in the above
// generic partial specialization. This also avoids the poor (50%) register
// utilization of FragmentUint8x4x1: by handling 16 scalar values at once,
// we are able to fill a uint8x16_t.
template <>
struct OutputStageEvalImpl<OutputStageSaturatingCastToUint8,
                           AVX2FragmentInt32x16x1> {
  typedef AVX2FragmentInt32x32x1 InputType;
  typedef AVX2FragmentUint8x32x1 OutputType;
  typedef OutputStageSaturatingCastToUint8 OutputStage;

  OutputStageEvalImpl(const OutputStage&) {}

  OutputType Eval(InputType input, int, int) const {
    __m256i q16[2];
    for (int i = 0; i < 2; i++) {
      q16[i] = _mm256_packus_epi32(input.data.val[2 * i],
				   input.data.val[2 * i + 1]);
    }
    return _mm256_packus_epi16(q16[0], q16[1]);
  }
};

// Implementation of OutputStageBiasAddition for AVX2FragmentInt32x8x1
template <typename VectorType>
struct OutputStageEvalImpl<OutputStageBiasAddition<VectorType>,
                           AVX2FragmentInt32x8x1> {
  typedef AVX2FragmentInt32x8x1 InputType;
  typedef AVX2FragmentInt32x8x1 OutputType;
  typedef OutputStageBiasAddition<VectorType> OutputStage;

  OutputStageEvalImpl(const OutputStage& s) : output_stage(s) {}

  OutputType Eval(InputType input, int row, int col) const {
    __m256i bias;
    if (VectorType::kShape == VectorShape::Row) {
      bias = _mm256_set1_epi32(output_stage.bias_vector(col));
    } else {
      bias = _mm256_lddqu_si256(reinterpret_cast<__m256i*>
			     (output_stage.bias_vector.data(row)));
    }
    return _mm256_add_epi32(input, bias);
  }

  const OutputStage& output_stage;
};

// Implementation of OutputStageClamp for AVX2FragmentInt32x8x1
template <>
struct OutputStageEvalImpl<OutputStageClamp, AVX2FragmentInt32x8x1> {
  typedef AVX2FragmentInt32x8x1 InputType;
  typedef AVX2FragmentInt32x8x1 OutputType;
  typedef OutputStageClamp OutputStage;

  OutputStageEvalImpl(const OutputStage& s) : output_stage(s) {}

  OutputType Eval(InputType input, int, int) const {
    const __m256i min = _mm256_set1_epi32(output_stage.min);
    const __m256i max = _mm256_set1_epi32(output_stage.max);
    return _mm256_min_epi32(_mm256_max_epi32(input, min), max);
  }

  const OutputStage& output_stage;
};

// Implementation of OutputStageTanh for AVX2FragmentInt32x8x1
template <>
struct OutputStageEvalImpl<OutputStageTanh, AVX2FragmentInt32x8x1>
    : OutputStageTanhEvalImpl<AVX2FragmentInt32x8x1> {
  OutputStageEvalImpl(const OutputStageTanh& output_stage)
      : OutputStageTanhEvalImpl(output_stage) {}
};

// Specialization of StoreFinalOutput for AVX2FragmentUint8x8x1.
template <typename DstType>
inline void StoreFinalOutput(AVX2FragmentUint8x8x1 value, DstType* dst, int row,
                             int col) {
  unsigned char *tmp = dst->data(row,col);
  for (int i=0; i<8; i++)
    tmp[i] = (value >> (i*8)) & 0xff;
 }

// Specialization of StoreFinalOutput for AVX2FragmentUint8x16x1.
template <typename DstType>
inline void StoreFinalOutput(AVX2FragmentUint8x32x1 value, DstType* dst,
                             int row, int col) {
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst->data(row, col)), value);
}

// Specialization of StoreFinalOutput for AVX2FragmentInt32x8x1, storing into
// a int32 destination.
template <typename DstType>
inline void StoreFinalOutput(AVX2FragmentInt32x8x1 value, DstType* dst, int row,
                             int col) {
  _mm256_storeu_si256(reinterpret_cast<__m256i*> (dst->data(row, col)), value);
}

// Specialization of StoreFinalOutput for AVX2FragmentInt32x16x1, storing into
// a int32 destination.
template <typename DstType>
inline void StoreFinalOutput(AVX2FragmentInt32x16x1 value, DstType* dst,
                             int row, int col) {
  for (int i = 0; i < 4; i++) {
    _mm256_storeu_si256( reinterpret_cast<__m256i*>(dst->data(row + 4*i, col)),
		      value.data.val[i]);
  }
}

}  // namespace gemmlowp

#endif  // GEMMLOWP_INTERNAL_OUTPUT_AVX2_H_
