# -*- coding: utf-8 -*-
from abc import ABC
from statistics import mean

import numpy as np
from torch import nn


class InfoModule(ABC, nn.Module):
    def count_parameters(self):
        return sum(int(np.prod(p.size())) for p in self.parameters())

    def grad_norm(self):
        return mean(float(p.norm().item()) for p in self.parameters())