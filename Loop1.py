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


def loop1(A,B):
    a=np.copy(A)
    b=np.copy(B)

    for j in range (1,a.shape[0]-1):
        for i in range (1,a.shape[1]-1):
            a[j,i] = a[j - 1,i] + b[j,i];
            b[j,i] = b[j,i]**2
        
    return a,b

x,y=loop1(A,B)

## decorators here

##Laptop/jetson nano optimized using numba:
@njit(nopython=True, parallel=True)#Using parallelism allows us to use all cores of laptop or jetson nano
def loop1_sol(A,B):
    a=np.copy(A)
    b=np.copy(B)

    for j in range (1,a.shape[0]-1):
        for i in range (1,a.shape[1]-1):
            a[j,i] = a[j - 1,i] + b[j,i];
            b[j,i] = b[j,i]**2
        
    return a,b

x1,y1=loop1_sol(A,B)

##### If output is false, then your solution is incorrect

print(np.allclose(x,x1))
print(np.allclose(y,y1))