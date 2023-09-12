"""
Test the batch normalisation switch layer
"""

import pytest
import torch

from augment_to_interpret.models import BatchNormSwitch, batchnorm_switch


def test_it_all():
    batchnorm_switch.use_old_version_of_the_code = False
    bns = BatchNormSwitch(10)
    with BatchNormSwitch.Switch(1):
        print("start")
        _ = bns(torch.tensor([[float(i) for i in range(10)], [10 - float(i) for i in range(10)]]))
        print("done")

    with pytest.raises(RuntimeError):
        _ = bns(torch.tensor([[float(i) for i in range(10)], [10 - float(i) for i in range(10)]]))
