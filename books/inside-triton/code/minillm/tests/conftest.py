import pytest
import torch

from minillm import DEVICE


@pytest.fixture(autouse=True)
def _seed():
    # 每个测试都从同一个随机种子开始，失败时能稳定复现
    torch.manual_seed(0)


@pytest.fixture
def device():
    return DEVICE
