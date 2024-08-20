#pragma once

#include "kai_pack.h"
#include "kai_ukernel_interface.h"

#if AT_KLEIDIAI_ENABLED()

namespace at::native::kleidiai {

/**
 * @brief Rearranges the quantized weight to support kleidiai inference
 * @param bl Groupsize for quantization. 32 for groupwise , 0 for channelwise
 */
Tensor kai_pack_rhs(
    const Tensor& weight_packed,
    const Tensor& weight,
    const Tensor& scales,
    const int64_t n,
    const int64_t k,
    const int64_t bl);

/**
 * @brief Outputs the buffer size for the packed weights
 * @param bl Groupsize for quantization. 32 for groupwise , 0 for channelwise
 */
size_t kai_pack_rhs_size(const int64_t n, const int64_t k, const int64_t bl);

/**
 * @brief Run 2 operations ( Input quantize and pack  + Matmul )
 */
void kai_quant_pack_lfs_mm_channelwise(
    const Tensor& output,
    const Tensor& input,
    const Tensor& weight,
    const int64_t m,
    const int64_t n,
    const int64_t k);

/**
 * @brief Run 2 operations ( Input quantize and pack  + Matmul )
 */
void kai_quant_pack_lfs_mm_groupwise(
    const Tensor& output,
    const Tensor& input,
    const Tensor& weight,
    const int64_t m,
    const int64_t n,
    const int64_t k,
    const int64_t bl);

} // namespace at::native::kleidiai
#endif
