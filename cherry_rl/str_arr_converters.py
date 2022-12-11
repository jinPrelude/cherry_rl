import numpy as np

def vector_gym_arr2str(array: np.array):
    return np.array2string(array)[2:-2]

def vector_gym_str2arr(string: str):
    return np.fromstring(string, dtype=float, sep=" ")