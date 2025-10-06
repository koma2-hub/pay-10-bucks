import os 
import numpy as np
import torch
import fpsample
from utils.data_utils import load_ply , knn

values = np.random.uniform(0, 20, 100)
print(values)

step_values = np.round(values/0.5)
print(step_values)

min_val = -0.1
max_val = 0.1
step = 0.01

num_steps = int((max_val - min_val) / step) + 1
random_integer = np.random.randint(0, num_steps, 10)
result = min_val + random_integer*step
print(result)

def generate_noise(min, max, step, num):
    num_steps = int((max - min) / step) + 1
    random_integer = np.random.randint(0, num_steps, num)
    result = min + random_integer*step
    return result

print(generate_noise(-0.1, 0.1, 0.01, 32))

print((np.random.normal(0, 0.05, 32)))