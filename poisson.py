import ctypes
import numpy as np

lib = ctypes.CDLL("poisson.o")

__c_poisson = lib.poisson
__c_poisson.restype = ctypes.c_double
__c_poisson.argtypes = [
    np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), 
    np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    ctypes.c_int,
    ctypes.c_int
]

def poisson(f: np.ndarray[np.double], h: np.ndarray[np.double], width: int, height: int) -> float:
    return __c_poisson(f, h, width, height)