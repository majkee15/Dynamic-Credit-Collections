#The only test

import dcc
from dcc import Parameters
import numpy as np

balance = 75
params = Parameters()
aavcl = dcc.AAV(params)
print(aavcl.compute_w0star())
w_array = np.linspace(0, 100, 40)
l_array = np.linspace(0, 2, 10)
aavcl.evaluate_aav(1, w_array, True)
aavcl.evaluate_aav(l_array, w_array, True)
aavcl.evaluate_aav(l_array, balance, True)