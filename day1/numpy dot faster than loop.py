import numpy as np
import time

np.random.seed(1)

def dotproduct(a,b):
    #parameters are two arrays of same dimension
    
    x = 0
    for i in range(a.shape[0]):
        x = x+a[i]*b[i]
    return x


a = np.random.rand(1000000)
b = np.random.rand(1000000)

tic = time.time() #start time
c = np.dot(a,b)
toc = time.time() #end time

print(f"dot(a,b) numpy = {c:.4f}")
print(f"time taken - {1000*(toc-tic):.4f} ms")


tic = time.time() #start time
c = dotproduct(a, b)
toc = time.time() #end time


print(f"dot(a,b) = {c:.4f}")
print(f"time taken - {1000*(toc-tic):.4f} ms")


del(a);
del(b);