#!python
#cython: language_level=3
import numpy as np
from libc.math cimport abs
cimport cython


ctypedef fused floating:
   double
   float


@cython.boundscheck(False)
@cython.wraparound(False)
def _enforce_temporal_decrease(
    floating[:,:] temporal_traces,
    const long[:] peak_times,
):
    cdef Py_ssize_t M = temporal_traces.shape[0]
    cdef Py_ssize_t T = temporal_traces.shape[1]
    cdef int j, p 
    cdef floating prev, cur

    for j in range(M):
        p = peak_times[j]
        prev = abs(temporal_traces[j, p])
        for t in range(p, T):
            cur = abs(temporal_traces[j, t])
            if cur > prev and cur > 0:
                temporal_traces[j, t] *= prev / cur
