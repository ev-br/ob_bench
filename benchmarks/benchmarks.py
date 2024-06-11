# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.

'''
class TimeSuite:
    """
    An example benchmark that times the performance of various kinds
    of iterating over dictionaries in Python.
    """
    def setup(self):
        self.d = {}
        for x in range(500):
            self.d[x] = None

    def time_keys(self):
        for key in self.d.keys():
            pass

    def time_values(self):
        for value in self.d.values():
            pass

    def time_range(self):
        d = self.d
        for key in range(500):
            d[key]


class MemSuite:
    def mem_list(self):
        return [0] * 256
'''


import numpy as np
from openblas_wrap import (
    # level 1
    dnrm2, ddot, daxpy,
    # level 3
    dgemm, dsyrk,
    # lapack
    dgesv,                   # linalg.solve
    dgesdd, dgesdd_lwork,    # linalg.svd
    dsyev, dsyev_lwork,      # linalg.eigh
)

# ### BLAS level 1 ###

# dnrm2

dnrm2_sizes = [100, 1000]

def run_dnrm2(n, x, incx):
    res = dnrm2(x, n, incx=incx)
    return res



class Nrm2:

    params = [100, 1000]
    param_names = ["size"]

    def setup(self, n):
        rndm = np.random.RandomState(1234)
        self.x = rndm.uniform(size=(n,)).astype(float)

    def time_dnrm2(self, n):
        run_dnrm2(n, self.x, 1)

'''
dnrm2_sizes = [100, 1000]

def run_dnrm2(n, x, incx):
    res = dnrm2(x, n, incx=incx)
    return res



def time_single_run():
    n = 10
    import time
    time.sleep(1)

   # run_dnrm2(n)

def time_single_run_2():
    n = 100
    run_dnrm2(n)
'''


'''
@pytest.mark.parametrize('n', dnrm2_sizes)
def test_nrm2(benchmark, n):
    rndm = np.random.RandomState(1234)
    x = np.array(rndm.uniform(size=(n,)), dtype=float)
    result = benchmark(run_dnrm2, n, x, 1)
'''

