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

// fixedpoint.h: fixed-point arithmetic, with basic operations and
// a few math functions such as tanh.

#ifndef GEMMLOWP_INTERNAL_FIXEDPOINT_H_
#define GEMMLOWP_INTERNAL_FIXEDPOINT_H_

#include <cassert>
#include <limits>

#include "../internal/common.h"

namespace gemmlowp {

// Part 1: Low-level integer-arithmetic primitives.
// The implementations here are generic implementations valid for
// scalar types (e.g. int32_t). Architecture-specific SIMD types
// (e.g. NEON int32x4_t) may be supported by providing
// specializations for them in separate files.
//
// The purpose of these primitives is two-fold:
//  - They will be used to implement higher-level fixed-point
//    abstractions, namely the FixedPoint class and its arithmetic
//    operators.
//  - They will be directly used to implement some more involved
//    fixed-point computations, e.g. the fixed-point implementation
//    of math functions such as tanh.

// Plain bit-wise AND
template <typename tIntegerType>
tIntegerType BitAnd(tIntegerType a, tIntegerType b) {
  return a & b;
}

// Plain bit-wise OR
template <typename tIntegerType>
tIntegerType BitOr(tIntegerType a, tIntegerType b) {
  return a | b;
}

// Plain bit-wise XOR
template <typename tIntegerType>
tIntegerType BitXor(tIntegerType a, tIntegerType b) {
  return a ^ b;
}

// Plain bit-wise NOT
template <typename tIntegerType>
tIntegerType BitNot(tIntegerType a) {
  return ~a;
}

// Integer addition. Not saturating. Overflow is undefined behavior.
template <typename tIntegerType>
tIntegerType Add(tIntegerType a, tIntegerType b) {
  return a + b;
}

// Integer subtraction. Not saturating. Overflow is undefined behavior.
template <typename tIntegerType>
tIntegerType Mul(tIntegerType a, tIntegerType b) {
  return a * b;
}

 
template <typename tIntegerType>
tIntegerType Sub(tIntegerType a, tIntegerType b) {
  return a - b;
}

// Integer unary negative. Not saturating. Overflow is undefined behavior.
template <typename tIntegerType>
tIntegerType Neg(tIntegerType a) {
  return -a;
}

// Integer arithmetic left-shift, equivalent to multiplying with a
// power of two. Not saturating. Overflow is undefined behavior.
template <typename tIntegerType>
tIntegerType ShiftLeft(tIntegerType a, int offset) {
  return a * (1 << offset);
}

// Integer arithmetic right-shift, equivalent to dividing by a
// power of two. Rounds towards zero, like an ordinary integer division.
template <typename tIntegerType>
tIntegerType ShiftRight(tIntegerType a, int offset) {
  return a / (1 << offset);
}

// Each bit of the result is set to the corresponding bit of either then_val or
// else_val depending on whether the corresponding bit of if_mask is set.
// Equivalent to the VBSL instruction in ARM NEON.
template <typename tIntegerType>
tIntegerType SelectUsingMask(tIntegerType if_mask, tIntegerType then_val,
                             tIntegerType else_val) {
  return BitXor(BitAnd(if_mask, then_val), BitAnd(BitNot(if_mask), else_val));
}

// For each input scalar, the corresponding bits of the result are set if the
// input scalar is non-zero.
template <typename tIntegerType>
tIntegerType MaskIfNonZero(tIntegerType a) {
  static const tIntegerType zero = 0;
  return a ? BitNot(zero) : zero;
}

// For each input scalar, the corresponding bits of the result are set if the
// input scalar is zero.
template <typename tIntegerType>
tIntegerType MaskIfZero(tIntegerType a) {
  return MaskIfNonZero<tIntegerType>(!a);
}

// For each pair of input scalars, the corresponding bits of the result are
// set if the input scalars are equal.
template <typename tIntegerType>
tIntegerType MaskIfEqual(tIntegerType a, tIntegerType b) {
  return MaskIfNonZero<tIntegerType>(a == b);
}

// For each pair of input scalars, the corresponding bits of the result are
// set if the input scalars are not equal.
template <typename tIntegerType>
tIntegerType MaskIfNotEqual(tIntegerType a, tIntegerType b) {
  return MaskIfNonZero<tIntegerType>(a != b);
}

// For each pair of input scalars, the corresponding bits of the result are
// set if the input scalars a, b satisfy a > b.
template <typename tIntegerType>
tIntegerType MaskIfGreaterThan(tIntegerType a, tIntegerType b) {
  return MaskIfNonZero<tIntegerType>(a > b);
}

// For each pair of input scalars, the corresponding bits of the result are
// set if the input scalars a, b satisfy a >= b.
template <typename tIntegerType>
tIntegerType MaskIfGreaterThanOrEqual(tIntegerType a, tIntegerType b) {
  return MaskIfNonZero<tIntegerType>(a >= b);
}

// For each pair of input scalars, the corresponding bits of the result are
// set if the input scalars a, b satisfy a < b.
template <typename tIntegerType>
tIntegerType MaskIfLessThan(tIntegerType a, tIntegerType b) {
  return MaskIfNonZero<tIntegerType>(a < b);
}

// For each pair of input scalars, the corresponding bits of the result are
// set if the input scalars a, b satisfy a <= b.
template <typename tIntegerType>
tIntegerType MaskIfLessThanOrEqual(tIntegerType a, tIntegerType b) {
  return MaskIfNonZero<tIntegerType>(a <= b);
}

// Returns true if all of the input scalars are nonzero.
// This function may currently assume that each of the input scalars has either
// all or none of its bits set. Otherwise, its behavior is currently undefined.
template <typename tIntegerType>
bool All(tIntegerType a) {
  return a;
}

// Returns true if any of the input scalars are nonzero.
// This function may currently assume that each of the input scalars has either
// all or none of its bits set. Otherwise, its behavior is currently undefined.
template <typename tIntegerType>
bool Any(tIntegerType a) {
  return a;
}

// Returns (a+b)/2, rounded to the nearest integer.
// Equivalent to VRHADD in the ARM NEON instruction set.
template <typename IntegerType>
IntegerType RoundingHalfSum(IntegerType a, IntegerType b) {
  static_assert(std::is_same<IntegerType, void>::value, "unimplemented");
  return a;
}

template <>
inline int32_t RoundingHalfSum(int32_t a, int32_t b) {
  int64_t a64 = a;
  int64_t b64 = b;
  int64_t sum = a64 + b64;
  int64_t sign = sum >= 0 ? 1 : -1;
  return static_cast<int32_t>((sum + sign) / 2);
}

// Returns the integer that represents the product of two fixed-point
// numbers, interpreting all integers as fixed-point values in the
// interval [-1, 1), rounding to the nearest value, and saturating
// -1 * -1 to the maximum value (since 1 is not in the half-open
// interval [-1, 1)).
//
// [The explanation below specializes to int32_t for example purpose.]
//
// The mapping between IntegerType and the interval [-1, 1) is unique and
// implied by IntegerType, which is assumed to be signed. For example,
// for IntergerType==int32_t, the mapping is
//   real_value = integer_value / 2^31.
// So in this case, and leaving aside rounding and saturating, this
// function computes ((a / 2^31) * (b / 2^31)) * 2^31, which simplifies to
//   (a * b) / 2^31.
//
// The 'doubling' part in the name of this function comes from the fact that
// this operation is very close to a "multiply-high" operation, keeping only
// the top half bits, except that that would be effectively computing
//   (a * b) / 2^32,
// so here we are computing 2x that, since
//   1/2^31 = 2 * 1/2^32.
// The idea is to use all of the available 32 bits in the destination int32
// value.
//
// [End of the explanation specializing to int32.]
//
// This is equivalent to the VQRDMULH instruction in ARM NEON.
template <typename IntegerType>
IntegerType SaturatingRoundingDoublingHighMul(IntegerType a, IntegerType b) {
  static_assert(std::is_same<IntegerType, void>::value, "unimplemented");
  return a;
}

// This function implements the same computation as the ARMv7 NEON VQRDMULH
// instruction.
template <>
inline int32_t SaturatingRoundingDoublingHighMul(int32_t a, int32_t b) {
  bool overflow = a == b && a == std::numeric_limits<int32_t>::min();
  int64_t a_64(a);
  int64_t b_64(b);
  int64_t ab_64 = a_64 * b_64;
  int32_t nudge = ab_64 >= 0 ? (1 << 30) : (1 - (1 << 30));
  int32_t ab_x2_high32 = static_cast<int32_t>((ab_64 + nudge) / (1ll << 31));
  return overflow ? std::numeric_limits<int32_t>::max() : ab_x2_high32;
}

// Returns the product of a run-time integer value by a compile-time power
// of two, with either a positive exponent (equivalent to an arithmetic
// left shift, saturating) or a negative exponent (equivalent to an arithmetic
// right shift, rounding to nearest).
template <int Exponent, typename IntegerType,
          int ExponentSign = (Exponent > 0 ? 1 : Exponent < 0 ? -1 : 0)>
struct ImplSaturatingRoundingMultiplyByPOT {};

template <int Exponent, typename IntegerType>
struct ImplSaturatingRoundingMultiplyByPOT<Exponent, IntegerType, 0> {
  static IntegerType eval(IntegerType x) { return x; }
};

template <int Exponent>
struct ImplSaturatingRoundingMultiplyByPOT<Exponent, int32_t, 1> {
  static int32_t eval(int32_t x) {
    const int64_t min = std::numeric_limits<int32_t>::min();
    const int64_t max = std::numeric_limits<int32_t>::max();
    return x >= (1 << (31 - Exponent))
               ? max
               : x <= -(1 << (31 - Exponent)) ? min : x * (1 << Exponent);
  }
};

template <int Exponent>
struct ImplSaturatingRoundingMultiplyByPOT<Exponent, int32_t, -1> {
  static int32_t eval(int32_t x) {
    int32_t b = (std::abs(x) & (1 << (-Exponent - 1))) >> (-Exponent - 1);
    int32_t nudge = x >= 0 ? b : -b;
    return x / (1 << -Exponent) + nudge;
  }
};

template <int Exponent, typename IntegerType>
IntegerType SaturatingRoundingMultiplyByPOT(IntegerType x) {
  return ImplSaturatingRoundingMultiplyByPOT<Exponent, IntegerType>::eval(x);
}

// Some compile-time traits around raw types to handle SIMD aspects:
// number of lanes, underlying scalar type.
template <typename tIntegerType>
struct FixedPointRawTypeTraits {};

template <>
struct FixedPointRawTypeTraits<int32_t> {
  typedef int32_t ScalarRawType;
  static const int kLanes = 1;
};

// Returns a SIMD value duplicating a scalar value across all lanes.
template <typename tRawType>
tRawType Dup(typename FixedPointRawTypeTraits<tRawType>::ScalarRawType x) {
  return x;
}

// Part 2: the FixedPoint class.

// A FixedPoint object represents a fixed-point value stored in the underlying
// integer type tRawType, if tRawType is a plain scalar integer type.
// Alternatively, tRawType may be a SIMD type (e.g. NEON int32x4_t) in which
// case a FixedPoint object represents a corresponding SIMD vector of fixed
// point values.
//
// tIntegerBits describes the range of the fixed-point format: if
// tIntegerBits == m then the range of representable values is the half-open
// interval [-2^m; 2^m) where the open boundary on the right side means that
// 2^m is not representable (how close the maximum representable value is to
// it, depends on bit-depth of tRawType).
//
// In "Q format notation",
//   https://en.wikipedia.org/wiki/Q_(number_format)
// we are describing the format
//   Qm.n
// where
//   m = tIntegerBits
// and
//   n = NumberOfBits(tRawType) - (m + 1)
// Note that the (m + 1) in the above line is because we adopt the convention
// that we count the integer bits exclusively of the sign bit; so (m + 1) is
// the total number of integer bits inclusive of the sign bit.
//
// Accordingly, the number of integral representable values in our range
//   [-2^m ; 2^m)
// is equal to 2^(m+1).
template <typename tRawType, int tIntegerBits>
class FixedPoint {
 public:
  typedef tRawType RawType;

  typedef FixedPointRawTypeTraits<RawType> RawTypeTraits;
  typedef typename RawTypeTraits::ScalarRawType ScalarRawType;

  static const int kTotalBits = 8 * sizeof(ScalarRawType);
  static const int kIntegerBits = tIntegerBits;
  static const int kFractionalBits = kTotalBits - 1 - kIntegerBits;
  static_assert(kIntegerBits >= 0 && kIntegerBits < kTotalBits,
                "bad IntegerBits");

  typedef FixedPoint<ScalarRawType, kIntegerBits> ScalarFixedPointType;

  static const ScalarRawType ScalarRawMin() {
    return std::numeric_limits<ScalarRawType>::min();
  }

  static const ScalarRawType ScalarRawMax() {
    return std::numeric_limits<ScalarRawType>::max();
  }

  static const ScalarRawType RawMin() {
    return VectorFromScalar(ScalarRawMin());
  }

  static const ScalarRawType RawMax() {
    return VectorFromScalar(ScalarRawMax());
  }

  static FixedPoint FromRaw(RawType x) {
    FixedPoint retval;
    retval.raw() = x;
    return retval;
  }

  static FixedPoint FromScalarRaw(ScalarRawType x) {
    FixedPoint retval;
    retval.raw() = Dup<RawType>(x);
    return retval;
  }

  static FixedPoint FromScalarFixedPoint(ScalarFixedPointType x) {
    return FromScalarRaw(x.raw());
  }

  template <int Exponent>
  static FixedPoint ConstantPOT() {
    static const int kOffset = kFractionalBits + Exponent;
    static_assert(
        kOffset < 31,
        "Constant not exactly representable in this fixed-point format");
    return FromScalarRaw(ScalarRawType(1) << kOffset);
  }

  static FixedPoint Zero() { return FromScalarRaw(0); }

  static FixedPoint One() {
    return FromScalarRaw(kIntegerBits == 0
                             ? ScalarRawMax()
                             : (ScalarRawType(1) << kFractionalBits));
  }

  static FixedPoint FromDouble(double x) {
    const double min_bound = static_cast<double>(ScalarRawMin());
    const double max_bound = static_cast<double>(ScalarRawMax());
    return FromScalarRaw(static_cast<int32_t>(std::min(
        std::max(round(x * static_cast<double>(1ll << kFractionalBits)),
                 min_bound),
        max_bound)));
  }

  RawType raw() const { return i_; }
  RawType& raw() { return i_; }

 private:
  RawType i_;
};

// Part 3: implementation of arithmetic operators for the
// FixedPoint class, and a few related functions.

// A FixedPoint multiplication is just a
// SaturatingRoundingDoublingHighMul operation on the underlying
// raw integer values. The IntegerBits simply add up, as is obvious
// from the fact that the range is [-2^IntegerBits, 2^IntegerBits).
template <typename tRawType, int tIntegerBits_a, int tIntegerBits_b>
FixedPoint<tRawType, tIntegerBits_a + tIntegerBits_b> operator*(
    FixedPoint<tRawType, tIntegerBits_a> a,
    FixedPoint<tRawType, tIntegerBits_b> b) {
  FixedPoint<tRawType, tIntegerBits_a + tIntegerBits_b> c;
  c.raw() = SaturatingRoundingDoublingHighMul(a.raw(), b.raw());
  return c;
}

// Tweaking IntegerBits gives exact multiplication by a power of two.
template <int tExponent, typename tRawType, int tIntegerBits>
FixedPoint<tRawType, tExponent + tIntegerBits> ExactMulByPot(
    FixedPoint<tRawType, tIntegerBits> a) {
  FixedPoint<tRawType, tExponent + tIntegerBits> c;
  c.raw() = a.raw();
  return c;
}

// If we want to leave IntegerBits fixed, then multiplication
// by a power of two has to be saturating/rounding, not exact anymore.
template <int tExponent, typename tRawType, int tIntegerBits>
FixedPoint<tRawType, tIntegerBits> SaturatingRoundingMultiplyByPOT(
    FixedPoint<tRawType, tIntegerBits> a) {
  return FixedPoint<tRawType, tIntegerBits>::FromRaw(
      SaturatingRoundingMultiplyByPOT<tExponent>(a.raw()));
}

// Generic arithmetic operators.

#define MAKE_FIXEDPOINT_UNARY_FUNC(FuncName, ImplFuncName)                     \
  template <typename tRawType, int tIntegerBits>                               \
  FixedPoint<tRawType, tIntegerBits> FuncName(                                 \
      FixedPoint<tRawType, tIntegerBits> a) {                                  \
    return FixedPoint<tRawType, tIntegerBits>::FromRaw(ImplFuncName(a.raw())); \
  }

#define MAKE_FIXEDPOINT_BINARY_FUNC(FuncName, ImplFuncName) \
  template <typename tRawType, int tIntegerBits>            \
  FixedPoint<tRawType, tIntegerBits> FuncName(              \
      FixedPoint<tRawType, tIntegerBits> a,                 \
      FixedPoint<tRawType, tIntegerBits> b) {               \
    return FixedPoint<tRawType, tIntegerBits>::FromRaw(     \
        ImplFuncName(a.raw(), b.raw()));                    \
  }

MAKE_FIXEDPOINT_UNARY_FUNC(operator-, Neg)
MAKE_FIXEDPOINT_UNARY_FUNC(operator~, BitNot)
MAKE_FIXEDPOINT_BINARY_FUNC(operator+, Add)
MAKE_FIXEDPOINT_BINARY_FUNC(operator-, Sub)
MAKE_FIXEDPOINT_BINARY_FUNC(operator&, BitAnd)
MAKE_FIXEDPOINT_BINARY_FUNC(operator^, BitXor)
MAKE_FIXEDPOINT_BINARY_FUNC(operator|, BitOr)
MAKE_FIXEDPOINT_BINARY_FUNC(RoundingHalfSum, RoundingHalfSum)

#undef MAKE_FIXEDPOINT_UNARY_FUNC
#undef MAKE_FIXEDPOINT_BINARY_FUNC

#define MAKE_FIXEDPOINT_UNARY_FUNC_RETURNING_RAW(FuncName)  \
  template <typename tRawType, int tIntegerBits>            \
  tRawType FuncName(FixedPoint<tRawType, tIntegerBits> a) { \
    return FuncName(a.raw());                               \
  }

#define MAKE_FIXEDPOINT_BINARY_FUNC_RETURNING_RAW(FuncName) \
  template <typename tRawType, int tIntegerBits>            \
  tRawType FuncName(FixedPoint<tRawType, tIntegerBits> a,   \
                    FixedPoint<tRawType, tIntegerBits> b) { \
    return FuncName(a.raw(), b.raw());                      \
  }

MAKE_FIXEDPOINT_UNARY_FUNC_RETURNING_RAW(MaskIfZero)
MAKE_FIXEDPOINT_UNARY_FUNC_RETURNING_RAW(MaskIfNonZero)
MAKE_FIXEDPOINT_BINARY_FUNC_RETURNING_RAW(MaskIfEqual)
MAKE_FIXEDPOINT_BINARY_FUNC_RETURNING_RAW(MaskIfNotEqual)
MAKE_FIXEDPOINT_BINARY_FUNC_RETURNING_RAW(MaskIfGreaterThan)
MAKE_FIXEDPOINT_BINARY_FUNC_RETURNING_RAW(MaskIfGreaterThanOrEqual)
MAKE_FIXEDPOINT_BINARY_FUNC_RETURNING_RAW(MaskIfLessThan)
MAKE_FIXEDPOINT_BINARY_FUNC_RETURNING_RAW(MaskIfLessThanOrEqual)

#undef MAKE_FIXEDPOINT_UNARY_FUNC_RETURNING_RAW
#undef MAKE_FIXEDPOINT_BINARY_FUNC_RETURNING_RAW

template <typename tRawType, int tIntegerBits>
FixedPoint<tRawType, tIntegerBits> SelectUsingMask(
    tRawType if_mask, FixedPoint<tRawType, tIntegerBits> then_val,
    FixedPoint<tRawType, tIntegerBits> else_val) {
  return FixedPoint<tRawType, tIntegerBits>::FromRaw(
      SelectUsingMask(if_mask, then_val.raw(), else_val.raw()));
}

template <typename tRawType, int tIntegerBits>
bool operator==(FixedPoint<tRawType, tIntegerBits> a,
                FixedPoint<tRawType, tIntegerBits> b) {
  return All(MaskIfEqual(a.raw(), b.raw()));
}

template <typename tRawType, int tIntegerBits>
bool operator!=(FixedPoint<tRawType, tIntegerBits> a,
                FixedPoint<tRawType, tIntegerBits> b) {
  return !(a == b);
}

// Conversion to floating-point.
template <typename tRawType, int tIntegerBits>
double ToDouble(FixedPoint<tRawType, tIntegerBits> x) {
  static_assert(FixedPointRawTypeTraits<tRawType>::kLanes == 1,
                "not applicable to SIMD types");
  typedef FixedPoint<tRawType, tIntegerBits> F;
  return x.raw() / static_cast<double>(1ll << F::kFractionalBits);
}

// Rescale changes the number of IntegerBits and updates the underlying
// raw integer value accordingly.
template <int tIntegerBitsDst, typename tRawType, int tIntegerBitsSrc>
FixedPoint<tRawType, tIntegerBitsDst> Rescale(
    FixedPoint<tRawType, tIntegerBitsSrc> x) {
  static const int kExponent = tIntegerBitsSrc - tIntegerBitsDst;
  FixedPoint<tRawType, tIntegerBitsDst> result;
  result.raw() = SaturatingRoundingMultiplyByPOT<kExponent>(x.raw());
  return result;
}

// CheckedFixedPointConstant allows to specify fixed-point constants
// initialized as real numbers, in a way that does not compile floating-point
// arithmetic in production code, yet still checks agreement with the
// floating-point expressions when asserts are enabled.
#ifdef GEMMLOWP_ENABLE_FIXEDPOINT_CONSTANTS_CHECKS
template <typename FixedPointType>
FixedPointType CheckedFixedPointConstant(
    typename FixedPointType::ScalarRawType raw_value, double double_value) {
  typedef typename FixedPointType::RawType RawType;
  const FixedPointType result = FixedPointType::FromScalarRaw(raw_value);
  assert(result == FixedPointType::FromDouble(double_value));
  return result;
}
#define GEMMLOWP_CHECKED_FIXEDPOINT_CONSTANT(FixedPointType, ScalarRawValue, \
                                             DoubleValue)                    \
  (CheckedFixedPointConstant<FixedPointType>(ScalarRawValue, DoubleValue))

#else
#define GEMMLOWP_CHECKED_FIXEDPOINT_CONSTANT(FixedPointType, ScalarRawValue, \
                                             DoubleValue)                    \
  (FixedPointType::FromScalarRaw(ScalarRawValue))
#endif

// Implementation of exponential function.

// Returns exp(x) for x in [-1/4, 0).
template <typename tRawType>
FixedPoint<tRawType, 0> exp_on_interval_between_negative_one_quarter_and_0_excl(
    FixedPoint<tRawType, 0> a) {
  typedef FixedPoint<tRawType, 0> F;
  const F constant_term =
      GEMMLOWP_CHECKED_FIXEDPOINT_CONSTANT(F, 1895147668, std::exp(-1.0 / 8.0));
  const F constant_1_over_3 =
      GEMMLOWP_CHECKED_FIXEDPOINT_CONSTANT(F, 715827883, 1.0 / 3.0);
  // We're evaluating a Taylor expansion around -1/8, so we do the change of
  // variable: x = a + 1/8.
  // In fixed-point with 0 integer bits, 1/8 is represented by 1 << 28.
  F x = a + F::template ConstantPOT<-3>();
  F x2 = x * x;
  F x3 = x2 * x;
  F x4 = x2 * x2;
  F x4_over_4 = SaturatingRoundingMultiplyByPOT<-2>(x4);
  F x4_over_24_plus_x3_over_6_plus_x2_over_2 =
      SaturatingRoundingMultiplyByPOT<-1>(
          ((x4_over_4 + x3) * constant_1_over_3) + x2);
  return constant_term +
         constant_term * (x + x4_over_24_plus_x3_over_6_plus_x2_over_2);
}

// Returns exp(x) for x < 0.
template <typename tRawType, int tIntegerBits>
FixedPoint<tRawType, 0> exp_on_negative_values(
    FixedPoint<tRawType, tIntegerBits> a) {
  typedef FixedPoint<tRawType, tIntegerBits> InputF;
  typedef FixedPoint<tRawType, 0> ResultF;
  static const int kFractionalBits = InputF::kFractionalBits;
  static const int kIntegerBits = InputF::kIntegerBits;
  static const InputF kOneQuarter = InputF::template ConstantPOT<-2>();
  InputF mask = kOneQuarter - InputF::FromScalarRaw(1);
  InputF a_mod_quarter_minus_one_quarter = (a & mask) - kOneQuarter;
  ResultF result = exp_on_interval_between_negative_one_quarter_and_0_excl(
      Rescale<0>(a_mod_quarter_minus_one_quarter));
  tRawType remainder = (a_mod_quarter_minus_one_quarter - a).raw();

#define GEMMLOWP_EXP_BARREL_SHIFTER(Exponent, FixedPointMultiplier)         \
  if (kIntegerBits > Exponent) {                                            \
    const ResultF kMultiplier = GEMMLOWP_CHECKED_FIXEDPOINT_CONSTANT(       \
        ResultF, FixedPointMultiplier, std::exp(-std::pow(2.0, Exponent))); \
    static constexpr int kShiftAmount =                                     \
        kIntegerBits > Exponent ? kFractionalBits + Exponent : 0;           \
    result = SelectUsingMask(                                               \
        MaskIfNonZero(BitAnd(remainder, Dup<tRawType>(1 << kShiftAmount))), \
        result * kMultiplier, result);                                      \
  }

  GEMMLOWP_EXP_BARREL_SHIFTER(-2, 1672461947);
  GEMMLOWP_EXP_BARREL_SHIFTER(-1, 1302514674);
  GEMMLOWP_EXP_BARREL_SHIFTER(+0, 790015084);
  GEMMLOWP_EXP_BARREL_SHIFTER(+1, 290630308);
  GEMMLOWP_EXP_BARREL_SHIFTER(+2, 39332535);
  GEMMLOWP_EXP_BARREL_SHIFTER(+3, 720401);
  GEMMLOWP_EXP_BARREL_SHIFTER(+4, 242);

#undef GEMMLOWP_EXP_BARREL_SHIFTER

  if (kIntegerBits > 5) {
    static const int b = kIntegerBits > 5 ? kFractionalBits + 5 : 0;
    const InputF clamp =
        GEMMLOWP_CHECKED_FIXEDPOINT_CONSTANT(InputF, -(1 << b), -32.0);
    result = SelectUsingMask(MaskIfLessThan(a, clamp), ResultF::Zero(), result);
  }

  result = SelectUsingMask(MaskIfZero(a), ResultF::One(), result);
  return result;
}

// Implementation of tanh: (1 - exp(-2x)) / (1 + exp(-2x)).

// Returns (1 - x) / (1 + x) for x in (0, 1).
template <typename tRawType>
FixedPoint<tRawType, 0> one_minus_x_over_one_plus_x_for_x_in_0_1(
    FixedPoint<tRawType, 0> a) {
  typedef FixedPoint<tRawType, 0> F0;
  typedef FixedPoint<tRawType, 2> F2;
  F0 half_denominator = RoundingHalfSum(a, F0::One());
  // Newton-Raphson division
  // https://en.wikipedia.org/wiki/Division_algorithm#Newton.E2.80.93Raphson_division
  // Refer to that page for the logic behind the 48/17 and 32/17 constants.
  const F2 constant_48_over_17 =
      GEMMLOWP_CHECKED_FIXEDPOINT_CONSTANT(F2, 1515870810, 48.0 / 17.0);
  const F2 constant_neg_32_over_17 =
      GEMMLOWP_CHECKED_FIXEDPOINT_CONSTANT(F2, -1010580540, -32.0 / 17.0);
  F2 x = constant_48_over_17 + half_denominator * constant_neg_32_over_17;
  for (int i = 0; i < 3; i++) {
    F2 half_denominator_times_x = half_denominator * x;
    F2 one_minus_half_denominator_times_x =
        F2::One() - half_denominator_times_x;
    x = x + Rescale<2>(x * one_minus_half_denominator_times_x);
  }
  return Rescale<0>(x - F2::One());
}

// Returns -tanh(x) for x < 0.
template <typename tRawType, int tIntegerBits>
FixedPoint<tRawType, 0> neg_tanh_on_negative_values(
    FixedPoint<tRawType, tIntegerBits> a) {
  return one_minus_x_over_one_plus_x_for_x_in_0_1(
      exp_on_negative_values(ExactMulByPot<1>(a)));
}

// Returns tanh(x) for any x.
template <typename tRawType, int tIntegerBits>
FixedPoint<tRawType, 0> tanh(FixedPoint<tRawType, tIntegerBits> a) {
  typedef FixedPoint<tRawType, tIntegerBits> InputF;
  typedef FixedPoint<tRawType, 0> ResultF;
  tRawType mask_if_negative = MaskIfLessThan(a, InputF::Zero());
  tRawType mask_if_zero = MaskIfZero(a);
  InputF n = SelectUsingMask(mask_if_negative, a, -a);
  ResultF t = neg_tanh_on_negative_values(n);
  return SelectUsingMask(mask_if_zero, ResultF::Zero(),
                         SelectUsingMask(mask_if_negative, -t, t));
}

// Implementation of logistic function.

// Returns 1 / (1 + x) for x in (0, 1).
template <typename tRawType>
FixedPoint<tRawType, 0> one_over_one_plus_x_for_x_in_0_1(
    FixedPoint<tRawType, 0> a) {
  typedef FixedPoint<tRawType, 0> F0;
  typedef FixedPoint<tRawType, 2> F2;
  F0 half_denominator = RoundingHalfSum(a, F0::One());
  // Newton-Raphson division
  // https://en.wikipedia.org/wiki/Division_algorithm#Newton.E2.80.93Raphson_division
  // Refer to that page for the logic behind the 48/17 and 32/17 constants.
  const F2 constant_48_over_17 =
      GEMMLOWP_CHECKED_FIXEDPOINT_CONSTANT(F2, 1515870810, 48.0 / 17.0);
  const F2 constant_neg_32_over_17 =
      GEMMLOWP_CHECKED_FIXEDPOINT_CONSTANT(F2, -1010580540, -32.0 / 17.0);
  F2 x = constant_48_over_17 + half_denominator * constant_neg_32_over_17;
  for (int i = 0; i < 3; i++) {
    F2 half_denominator_times_x = half_denominator * x;
    F2 one_minus_half_denominator_times_x =
        F2::One() - half_denominator_times_x;
    x = x + Rescale<2>(x * one_minus_half_denominator_times_x);
  }
  return Rescale<0>(ExactMulByPot<-1>(x));
}

// Returns logistic(x) = 1 / (1 + exp(-x)) for x > 0.
template <typename tRawType, int tIntegerBits>
FixedPoint<tRawType, 0> logistic_on_positive_values(
    FixedPoint<tRawType, tIntegerBits> a) {
  return one_over_one_plus_x_for_x_in_0_1(exp_on_negative_values(-a));
}

// Returns logistic(x) = 1 / (1 + exp(-x)) for any x.
template <typename tRawType, int tIntegerBits>
FixedPoint<tRawType, 0> logistic(FixedPoint<tRawType, tIntegerBits> a) {
  typedef FixedPoint<tRawType, tIntegerBits> InputF;
  typedef FixedPoint<tRawType, 0> ResultF;
  tRawType mask_if_positive = MaskIfGreaterThan(a, InputF::Zero());
  tRawType mask_if_zero = MaskIfZero(a);
  InputF abs_input = SelectUsingMask(mask_if_positive, a, -a);
  ResultF result_if_positive = logistic_on_positive_values(abs_input);
  ResultF result_if_negative = ResultF::One() - result_if_positive;
  const ResultF one_half =
      GEMMLOWP_CHECKED_FIXEDPOINT_CONSTANT(ResultF, 1 << 30, 0.5);
  return SelectUsingMask(mask_if_zero, one_half,
                         SelectUsingMask(mask_if_positive, result_if_positive,
                                         result_if_negative));
}

}  // end namespace gemmlowp

#ifdef GEMMLOWP_NEON
#include "./fixedpoint_neon.h"
#elif defined(GEMMLOWP_AVX2)
#include "./fixedpoint_avx2.h"
#elif defined(GEMMLOWP_SSE4)
#include "./fixedpoint_sse.h"
#endif

#endif  // GEMMLOWP_INTERNAL_FIXEDPOINT_H_
