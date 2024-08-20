# Owner(s): ["module: linear algebra"]

import torch
import numpy as np

import unittest
import itertools
import warnings
import math
from math import inf, nan, isnan
import re
import random
from random import randrange
from itertools import product
from functools import reduce, partial

from torch.testing._internal.common_utils import \
    (TestCase, run_tests, TEST_SCIPY, IS_MACOS, IS_WINDOWS, slowTest,
     TEST_WITH_ROCM, IS_FBCODE, IS_REMOTE_GPU, iter_indices,
     make_fullrank_matrices_with_distinct_singular_values,
     freeze_rng_state, IS_ARM64, IS_SANDCASTLE, TEST_OPT_EINSUM, parametrize, skipIfTorchDynamo,
     setBlasBackendsToDefaultFinally, setLinalgBackendsToDefaultFinally, serialTest,
     skipIfNoKleidiAI)
from torch.testing._internal.common_device_type import \
    (instantiate_device_type_tests, dtypes, has_cusolver, has_hipsolver,
     onlyCPU, skipCUDAIf, skipCUDAIfNoMagma, skipCPUIfNoLapack, precisionOverride,
     skipCUDAIfNoMagmaAndNoCusolver, skipCUDAIfRocm, onlyNativeDeviceTypes, dtypesIfCUDA,
     onlyCUDA, skipCUDAVersionIn, skipMeta, skipCUDAIfNoCusolver, skipCUDAIfNotRocm,
     dtypesIfMPS, largeTensorTest)

from torch.testing._internal.common_quantization import _group_quantize_tensor, _dynamically_quantize_per_channel, \
    _group_quantize_tensor_symmetric


class TestKleidiAI(TestCase):
    def setUp(self):
        super(self.__class__, self).setUp()
        torch.backends.cuda.matmul.allow_tf32 = False

    def tearDown(self):
        torch.backends.cuda.matmul.allow_tf32 = True
        super(self.__class__, self).tearDown()

    exact_dtype = True

    @skipIfNoKleidiAI
    @onlyCPU
    @parametrize("m", [1, 32])
    @parametrize("k", [128, 256, 4096])
    @parametrize("n", [512, 1024, 4096])
    def test__kai_int4_mm_channelwise(self, device, m, k, n):
        torch.manual_seed(1)
        input = torch.rand((m, k), dtype=torch.float32, device=device)
        weight = torch.rand((n, k), dtype=torch.float32, device=device)

        def convert_weight_to_int4pack(weight):
            q_group = weight.shape[-1]
            weight_uint8, weight_scales_and_zeros = _group_quantize_tensor_symmetric(
                weight, n_bit=4, groupsize=q_group, scheme="symmetric_channelwise"
            )
            weight_int4pack = torch._kai_weight_pack_int4(
                weight_uint8,
                weight_scales_and_zeros,
                weight.shape[-2],
                weight.shape[-1],
                0,
            )

            return weight_int4pack

        def weight_int4pack_mm(input, weight_int4pack, weight):
            return torch._kai_input_quant_mm_int4(
                input,
                weight_int4pack,
                input.shape[-2],
                weight.shape[-2],
                weight.shape[-1],
                0,
            )

        weight_int4pack = convert_weight_to_int4pack(weight)
        res = weight_int4pack_mm(input, weight_int4pack, weight)
        ref = torch.mm(input, weight.transpose(0, 1))

        mean_err = ((res - ref).abs() / ref).mean()
        self.assertTrue(mean_err < 0.05)

    @skipIfNoKleidiAI
    @onlyCPU
    @parametrize("m", [1, 32])
    @parametrize("k", [128, 256, 4096])
    @parametrize("n", [512, 1024, 4096])
    def test__kai_int4_mm_groupwise(self, device, m, k, n):
        torch.manual_seed(1)
        input = torch.rand((m, k), dtype=torch.float32, device=device)
        weight = torch.rand((n, k), dtype=torch.float32, device=device)
        q_group = 32

        def convert_weight_to_int4pack(weight):
            weight_uint8, weight_scales_and_zeros = _group_quantize_tensor_symmetric(
                weight, n_bit=4, groupsize=q_group, scheme="symmetric_groupwise"
            )
            weight_int4pack = torch._kai_weight_pack_int4(
                weight_uint8,
                weight_scales_and_zeros,
                weight.shape[-2],
                weight.shape[-1],
                q_group,
            )

            return weight_int4pack

        def weight_int4pack_mm(input, weight_int4pack, weight):
            return torch._kai_input_quant_mm_int4(
                input,
                weight_int4pack,
                input.shape[-2],
                weight.shape[-2],
                weight.shape[-1],
                q_group,
            )

        weight_int4pack = convert_weight_to_int4pack(weight)
        res = weight_int4pack_mm(input, weight_int4pack, weight)
        ref = torch.mm(input, weight.transpose(0, 1))

        mean_err = ((res - ref).abs() / ref).mean()
        self.assertTrue(mean_err < 0.05)


instantiate_device_type_tests(TestKleidiAI, globals())

if __name__ == "__main__":
    TestCase._default_dtype_check_enabled = True
    run_tests()
