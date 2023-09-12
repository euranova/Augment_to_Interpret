"""
Batch norm switch to avoid mixing distributions and comparable behaviour at train time and test time.
"""

import warnings

import torch

use_old_version_of_the_code = False  # Can be set to True even after initialisation of all modules.
_BATCH_NORM_N_SWITCHES = lambda: 10 if not use_old_version_of_the_code else 2

if use_old_version_of_the_code:
    warnings.warn("You are currently using the code WITHOUT the batchnorm fix.")


def default_bn_switch():
    return None if not use_old_version_of_the_code else 0


class BatchNormSwitch(torch.nn.Module):
    __batch_norm_switch = default_bn_switch

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.batch_norms = torch.nn.ModuleList([torch.nn.BatchNorm1d(*args, **kwargs)
                                                for _ in range(_BATCH_NORM_N_SWITCHES())])
        for switch in range(1, _BATCH_NORM_N_SWITCHES()):
            self.batch_norms[switch].bias, self.batch_norms[switch].weight = (
                self.batch_norms[0].bias, self.batch_norms[0].weight)

    def forward(self, *args, **kwargs):
        if BatchNormSwitch.__batch_norm_switch() not in list(range(_BATCH_NORM_N_SWITCHES())):
            if BatchNormSwitch.__batch_norm_switch() is None:
                raise RuntimeError(
                    "You cannot use a module using the batchnorm switch without explicitly setting "
                    "the value of the switch. To do so, whenever you call an intermediary function "
                    "of the module, such as model.clf.get_emb, ensure this called in wrapped in a  "
                    "'with BatchNormSwitch.Switch(value):' context.")
            raise RuntimeError(
                f"Batch norm switch can only be between 0 and {_BATCH_NORM_N_SWITCHES() - 1} "
                f"inclusive, but you entered {BatchNormSwitch.__batch_norm_switch()}. "
                f"Please use accepted values or modify BatchNormSwitch code.")
        return self.batch_norms[BatchNormSwitch.__batch_norm_switch()](*args, **kwargs)

    class Switch:
        def __init__(self, value):
            self.value = value

        def __enter__(self):
            if use_old_version_of_the_code:
                return
            if BatchNormSwitch._BatchNormSwitch__batch_norm_switch() is not None:
                raise RuntimeError(
                    f"The switch is already fixed to the value "
                    f"{BatchNormSwitch._BatchNormSwitch__batch_norm_switch()}. "
                    f"Do not open a Switch context inside another Switch context.")
            BatchNormSwitch._BatchNormSwitch__batch_norm_switch = lambda: self.value

        def __exit__(self, exc_type, exc_val, exc_tb):
            if use_old_version_of_the_code:
                return
            BatchNormSwitch._BatchNormSwitch__batch_norm_switch = default_bn_switch
