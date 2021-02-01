#This code is an implementation of the homomorphic filter using a Butterworth high pass filter. The program takes an image as input and it returns the filtered image
#as output. It also shows the pixels distribution of input and output image, the size of the Butterworth filter and his plot.

import logging
import numpy as np
import cv2
import matplotlib.pyplot as plt

#C:\Users\Giacomo\Downloads\image.png
path=input('Inserisci path dell\'immagine: ')
img=cv2.imread(r'%s'%path)
image=np.double(cv2.cvtColor(img,cv2.COLOR_RGB2GRAY))
image_log=np.log(image+1)
image_fft=np.fft.fft2(image_log)
Do=200
n=2.0
P=image.shape[0]/2
Q=image.shape[1]/2
rL=0.5
rH=1.5
c=0.1
#Butterworth high-pass filter
U,V=np.meshgrid(range(image_fft.shape[0]),range(image_fft.shape[1]),sparse=False,indexing='ij')
D=(((U-P)**2+(V-Q)**2)).astype(float)
X=1/(1+(((c*D)/Do**2)**n))
H=1-X
H=((rH-rL)*H)+rL
H=np.fft.fftshift(H)
#Applying
print("Image processing....")
image_fft_filtered=H*image_fft
image_filtered=np.fft.ifft2(image_fft_filtered)
final_image=np.exp(np.real(image_filtered))-1
#Histograms and show
fig, ax=plt.subplots(nrows=2, ncols=3, figsize=(11,8))
plt.gray()

ax[0][0].imshow(image)
ax[0][0].axis('off')
ax[0][0].set_title('Original image')

ax[1][0].hist(image.ravel(),bins=256)
ax[1][0].set_xlabel('Pixel values')
ax[1][0].set_ylabel('Number of pixels')
ax[1][0].set_title('Brightness distribution')

ax[0][1].imshow(final_image)
ax[0][1].axis('off')
ax[0][1].set_title('Filtered image')

ax[1][1].hist(final_image.ravel(),bins=256)
ax[1][1].set_xlabel('Pixel values')
ax[1][1].set_ylabel('Number of pixels')
ax[1][1].set_title('Brightness distribution')

ax[0][2].imshow(abs(H))
ax[0][2].axis('off')
ax[0][2].set_title('Butterworth high-pass filter')

ax[1][2].plot(D[0],H[0])
ax[1][2].set_title('Plot of Butterworth high-pass filter ')

fig.tight_layout()

plt.show()





  
