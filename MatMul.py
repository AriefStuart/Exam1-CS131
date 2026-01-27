import numpy as np
import ctypes
import math
import time
from scipy import misc,ndimage
import matplotlib.pyplot as plt
from PIL import Image
from numba import jit,prange
import requests
from io import BytesIO
import cv2

#### Import your libraries here
from numba import njit, prange
import numpy as np

url1="http://getwallpapers.com/wallpaper/full/c/a/7/1235918-3000-x-3000-hd-wallpapers-3000x2000-for-hd-1080p.jpg"
url2="http://getwallpapers.com/wallpaper/full/c/4/1/1235927-3000-x-3000-hd-wallpapers-3000x2000-screen.jpg"
response1 = requests.get(url1)
response2 = requests.get(url2)

### These will be used as input arrays, they are all images transformed to numpy arrays
#A = Image.open('nature.jpg')
# convert image to numpy array
A = Image.open(BytesIO(response1.content))
A = np.asarray(A)/255
A = np.mean(A, axis=2)*255

#B = Image.open('sun.jpg')
# convert image to numpy array
B = Image.open(BytesIO(response2.content))
B = np.asarray(B)/255
B = np.mean(B, axis=2)*255
C=np.random.rand(A.shape[0],A.shape[1])*255



# SGEMM Form => C := alpha*A*B + beta*C,
# We create new variable because the images are too big
Am=np.random.rand(100,500)
Bm=np.random.rand(500,250)
Dm=np.random.rand(100,250)
# alpha and beta are constant floating point values, reuse them for your solution
alpha = 1.5
beta = 1.2 

def sgemm_manual (alpha,A, B, beta, D):

    d=np.copy(D)
    for i in range(d.shape[0]):
        for j in range(d.shape[1]):
            d[i,j] *= beta
        for k in range(A.shape[1]):
            for j in range(d.shape[1]):
                d[i,j] += alpha * A[i,k] * B[k,j]
    return d            

x=sgemm_manual (alpha,Am, Bm, beta, Dm) 

# SGEMM Form => C := alpha*A*B + beta*C,
# We create new variable because the images are too big
Am=np.random.rand(100,500)
Bm=np.random.rand(500,250)
Dm=np.random.rand(100,250)
# alpha and beta are constant floating point values, reuse them for your solution
alpha = 1.5
beta = 1.2 

@njit(parallel=True)
def sgemm_solution (alpha,A, B, beta, D):
   d=np.copy(D)
   for i in prange(d.shape[0]):
       for j in range(d.shape[1]):
           d[i,j] *= beta
       for k in range(A.shape[1]):
           for j in range(d.shape[1]):
               d[i,j] += alpha * A[i,k] * B[k,j]
   return d 

x1=sgemm_solution (alpha,Am, Bm, beta, Dm)

##### If output is false, then your solution is incorrect
print(np.allclose(x,x1))