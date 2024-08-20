#pragma once
#include <ATen/Config.h>
#include <unordered_map>
#if AT_KLEIDIAI_ENABLED()

#include <kai_lhs_quant_pack_qai8dxp_f32.h>
#include <kai_lhs_quant_pack_qsi8d32p_f32.h>
#include <kai_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod.h>
#include <kai_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod.h>
#include <kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm.h>
#include <kai_matmul_clamp_f32_qai8dxp_qsi4cxp_interface.h>
#include <kai_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod.h>
#include <kai_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_8x4x32_neon_i8mm.h>
#include <kai_matmul_clamp_f32_qsi8d32p_qsi4c32p_interface.h>
#include <kai_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0.h>
#include <kai_rhs_pack_nxk_qsi4cxp_qsu4cxs1s0.h>

namespace at::native::kleidiai {

enum class kai_kernel_id {
  matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod =
      0, // 32 Groupwise GEMV
  matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_8x4x32_neon_i8mm =
      1, // 32 Groupwise GEMM
  matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod =
      2, // Channelwise GEMV
  matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm =
      3 // Channelwise GEMM
};

// Channelwise Kernel mapping
struct kai_matmul_ukernel_f32_qa8dxp_qs4cxp {
  struct kai_matmul_clamp_f32_qai8dxp_qsi4cxp_ukernel ukernel;
  struct kai_rhs_pack_nxk_qsi4cxp_qsu4cxs1s0_params rhs_pack_params;
  size_t (*kai_get_lhs_packed_size)(
      size_t m,
      size_t k,
      size_t mr,
      size_t kr,
      size_t sr);
  size_t (*kai_get_rhs_packed_size)(
      size_t n,
      size_t k,
      size_t nr,
      size_t kr,
      size_t sr);
  void (*kai_run_lhs_quant_pack)(
      size_t m,
      size_t k,
      size_t mr,
      size_t kr,
      size_t sr,
      size_t m_idx_start,
      const float* lhs,
      size_t lhs_stride,
      void* lhs_packed);
  void (*kai_run_rhs_pack)(
      size_t num_groups,
      size_t n,
      size_t k,
      size_t nr,
      size_t kr,
      size_t sr,
      const uint8_t* rhs,
      const float* bias,
      const float* scale,
      void* rhs_packed,
      size_t extra_bytes,
      const struct kai_rhs_pack_nxk_qsi4cxp_qsu4cxs1s0_params* params);

  kai_matmul_ukernel_f32_qa8dxp_qs4cxp(
      const kai_matmul_clamp_f32_qai8dxp_qsi4cxp_ukernel& kernel)
      : ukernel(kernel),
        kai_get_lhs_packed_size(
            &kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32),
        kai_get_rhs_packed_size(
            &kai_get_rhs_packed_size_rhs_pack_nxk_qsi4cxp_qsu4cxs1s0),
        kai_run_lhs_quant_pack(&kai_run_lhs_quant_pack_qai8dxp_f32),
        kai_run_rhs_pack(&kai_run_rhs_pack_nxk_qsi4cxp_qsu4cxs1s0) {}
};

struct kai_matmul_ukernel_f32_qa8dxp_qs4cxp
kai_select_channelwise_matmul_ukernel(const kai_kernel_id id);

// Groupwise Kernel mapping
struct kai_matmul_ukernel_f32_qa8d32p_qs4c32p {
  struct kai_matmul_clamp_f32_qsi8d32p_qsi4c32p_ukernel ukernel;
  struct kai_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0_params rhs_pack_params;
  size_t (*kai_get_lhs_packed_size)(
      size_t m,
      size_t k,
      size_t bl,
      size_t mr,
      size_t kr,
      size_t sr);
  size_t (*kai_get_rhs_packed_size)(
      size_t n,
      size_t k,
      size_t nr,
      size_t kr,
      size_t bl);
  void (*kai_run_lhs_quant_pack)(
      size_t m,
      size_t k,
      size_t bl,
      size_t mr,
      size_t kr,
      size_t sr,
      size_t m_idx_start,
      const float* lhs,
      size_t lhs_stride,
      void* lhs_packed);
  void (*kai_run_rhs_pack)(
      size_t num_groups,
      size_t n,
      size_t k,
      size_t nr,
      size_t kr,
      size_t sr,
      size_t bl,
      const uint8_t* rhs,
      const float* bias,
      void* rhs_packed,
      size_t extra_bytes,
      const struct kai_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0_params*
          params);

  kai_matmul_ukernel_f32_qa8d32p_qs4c32p(
      const kai_matmul_clamp_f32_qsi8d32p_qsi4c32p_ukernel& kernel)
      : ukernel(kernel),
        kai_get_lhs_packed_size(
            &kai_get_lhs_packed_size_lhs_quant_pack_qsi8d32p_f32),
        kai_get_rhs_packed_size(
            &kai_get_rhs_packed_size_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0),
        kai_run_lhs_quant_pack(&kai_run_lhs_quant_pack_qsi8d32p_f32),
        kai_run_rhs_pack(&kai_run_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0) {}
};

struct kai_matmul_ukernel_f32_qa8d32p_qs4c32p
kai_select_groupwise_matmul_ukernel(const kai_kernel_id id);

} // namespace at::native::kleidiai
#endif
