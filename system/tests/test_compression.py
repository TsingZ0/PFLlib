import os
import sys
import unittest
from collections import OrderedDict

import torch

# Ensure system/ is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flcore.trainmodel.models import FedAvgCNN, Mclr_Logistic
from utils.compression import (
    calc_compression_stats,
    dequantize_model,
    dequantize_tensor,
    quantize_model,
    quantize_tensor,
    simulate_quantization,
)


class TestPFLlibCompression(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_quantize_dequantize_tensor_int4(self):
        for shape in [(64,), (32, 32), (16, 3, 3, 3), (9, 17, 7)]:
            tensor = torch.randn(shape, dtype=torch.float32)
            q_dict = quantize_tensor(tensor, nbits=4, block_size=32)
            recon = dequantize_tensor(q_dict)

            self.assertEqual(recon.shape, tensor.shape)
            self.assertEqual(recon.dtype, tensor.dtype)
            cos_sim = torch.nn.functional.cosine_similarity(
                tensor.flatten(), recon.flatten(), dim=0
            )
            self.assertGreater(cos_sim.item(), 0.98)

    def test_quantize_dequantize_tensor_int8(self):
        tensor = torch.randn(64, 64, dtype=torch.float32)
        q_dict = quantize_tensor(tensor, nbits=8, block_size=64)
        recon = dequantize_tensor(q_dict)

        self.assertEqual(recon.shape, tensor.shape)
        cos_sim = torch.nn.functional.cosine_similarity(
            tensor.flatten(), recon.flatten(), dim=0
        )
        self.assertGreater(cos_sim.item(), 0.999)

    def test_zero_tensor_handling(self):
        tensor = torch.zeros(128, dtype=torch.float32)
        q_dict = quantize_tensor(tensor, nbits=4, block_size=32)
        recon = dequantize_tensor(q_dict)

        self.assertTrue(torch.equal(tensor, recon))
        self.assertFalse(torch.isnan(recon).any())
        self.assertFalse(torch.isinf(recon).any())

    def test_empty_tensor(self):
        tensor = torch.empty((0, 8), dtype=torch.float32)
        q_dict = quantize_tensor(tensor, nbits=4, block_size=32)
        recon = dequantize_tensor(q_dict)

        self.assertEqual(recon.shape, (0, 8))
        self.assertEqual(recon.numel(), 0)

    def test_sub_byte_bit_packing(self):
        numel = 256
        tensor = torch.randn(numel, dtype=torch.float32)
        q_dict = quantize_tensor(tensor, nbits=4, block_size=64)

        # 256 4-bit nibbles should pack into exactly 128 uint8 bytes
        self.assertEqual(q_dict["data"].dtype, torch.uint8)
        self.assertEqual(q_dict["data"].numel(), numel // 2)

    def test_quantize_model_module(self):
        model = FedAvgCNN(in_features=1, num_classes=10, dim=1024)
        q_dict = quantize_model(model, nbits=4, block_size=64)

        # Load back into a fresh model
        target_model = FedAvgCNN(in_features=1, num_classes=10, dim=1024)
        dequantize_model(q_dict, target_model=target_model)

        for p_orig, p_target in zip(model.parameters(), target_model.parameters()):
            self.assertEqual(p_orig.shape, p_target.shape)
            cos = torch.nn.functional.cosine_similarity(
                p_orig.flatten(), p_target.flatten(), dim=0
            )
            self.assertGreater(cos.item(), 0.98)

    def test_quantize_model_list(self):
        params = [torch.randn(16, 8), torch.randn(16), torch.tensor([1, 2, 3])]
        q_list = quantize_model(params, nbits=4, block_size=16)
        restored = dequantize_model(q_list)

        self.assertEqual(len(restored), len(params))
        for p, r in zip(params, restored):
            self.assertEqual(p.shape, r.shape)
            if torch.is_floating_point(p):
                cos = torch.nn.functional.cosine_similarity(
                    p.flatten(), r.flatten(), dim=0
                )
                self.assertGreater(cos.item(), 0.98)
            else:
                self.assertTrue(torch.equal(p, r))

    def test_quantize_model_state_dict(self):
        state_dict = OrderedDict(
            {
                "fc.weight": torch.randn(32, 16),
                "fc.bias": torch.randn(32),
                "running_mean": torch.zeros(32),
                "num_batches": torch.tensor(5, dtype=torch.long),
            }
        )
        q_dict = quantize_model(state_dict, nbits=4, block_size=16)
        restored = dequantize_model(q_dict)

        for k in state_dict:
            self.assertEqual(restored[k].shape, state_dict[k].shape)
            if torch.is_floating_point(state_dict[k]):
                if state_dict[k].abs().sum() > 0:
                    cos = torch.nn.functional.cosine_similarity(
                        state_dict[k].flatten(), restored[k].flatten(), dim=0
                    )
                    self.assertGreater(cos.item(), 0.98)
            else:
                self.assertTrue(torch.equal(state_dict[k], restored[k]))

    def test_error_feedback_residuals(self):
        residuals = {}
        true_accum = torch.zeros(64)
        quant_accum = torch.zeros(64)

        for _ in range(10):
            delta = torch.randn(64) * 0.05
            true_accum += delta

            p_dict = OrderedDict({"weight": delta})
            q_dict = quantize_model(p_dict, nbits=4, block_size=32, residuals=residuals)
            recon = dequantize_model(q_dict)
            quant_accum += recon["weight"]

        drift = torch.norm(true_accum - quant_accum).item()
        self.assertLess(drift, 0.05)

    def test_simulate_quantization(self):
        model = Mclr_Logistic(input_dim=784, num_classes=10)
        simulated = simulate_quantization(model, nbits=4, block_size=64)

        for p_orig, p_sim in zip(model.parameters(), simulated.parameters()):
            self.assertEqual(p_orig.shape, p_sim.shape)
            cos = torch.nn.functional.cosine_similarity(
                p_orig.flatten(), p_sim.flatten(), dim=0
            )
            self.assertGreater(cos.item(), 0.98)

    def test_calc_compression_stats(self):
        model = FedAvgCNN(in_features=1, num_classes=10, dim=1024)
        q_dict = quantize_model(model, nbits=4, block_size=64)
        stats = calc_compression_stats(model, q_dict)

        self.assertGreater(stats["compression_ratio"], 6.0)
        self.assertGreater(stats["bandwidth_savings_pct"], 80.0)
        self.assertGreater(stats["raw_bytes"], stats["compressed_bytes"])

    def test_invalid_inputs(self):
        tensor = torch.randn(32)
        with self.assertRaises(ValueError):
            quantize_tensor(tensor, nbits=6)
        with self.assertRaises(ValueError):
            quantize_tensor(tensor, nbits=4, block_size=5)


if __name__ == "__main__":
    unittest.main()
