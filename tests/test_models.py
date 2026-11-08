import unittest
import torch
from moire_project import repair, joint, spatial


class ModelContracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(2026)

    def test_zero_initialization_and_gradients(self):
        x, o, b = [torch.rand(1, 3, 32, 32) for _ in range(3)]
        for domain in ['DCID', 'UHDM']:
            model = repair.IndependentRepair(domain)
            y, field = model(x, o, b)
            self.assertTrue(torch.equal(y, repair.base_for_domain(domain, x, o, b)))
            self.assertEqual(field.abs().max().item(), 0)
            ((y - x).square().mean() + 0.001 * spatial.tv(field)).backward()
            self.assertGreater(model.net.final.weight.grad.abs().sum().item(), 0)
            self.assertEqual(sum(p.numel() for p in model.parameters()), 59883)

    def test_equal_candidates_preserved_by_fusion(self):
        x = torch.rand(1, 3, 32, 32)
        candidates = x[:, None].repeat(1, 3, 1, 1, 1)
        lows = torch.stack([joint.fusion_lowpass(x)] * 3, dim=1)
        net = joint.ScaleFusion('band', 2)
        output = net(x, candidates, lows)
        if isinstance(output, tuple):
            output = output[0]
        torch.testing.assert_close(output, x, atol=1e-6, rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
