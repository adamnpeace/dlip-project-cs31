import os
import skimage.io as io
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import skimage
from matplotlib import pyplot as plt
from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
import numpy as np
import time

def calculatePSNR(y_true,y_pred):

    psnr = skimage.measure.compare_psnr(y_true,y_pred)
    return psnr

def calculateMSE(y_true,y_pred):
    mse = skimage.measure.compare_mse(y_true,y_pred)
    return mse


def calculateNRMSE(y_true,y_pred):
    nrmse = skimage.measure.compare_nrmse(y_true,y_pred)
    return nrmse

def calculateSSIM(y_true,y_pred):
    ssim = skimage.measure.compare_ssim(y_true.reshape(128,128),y_pred.reshape(128,128))
    return ssim

def calculateSSIMWang(y_true,y_pred):
    return skimage.measure.compare_ssim(
        y_true.reshape(128, 128), 
        y_pred.reshape(128, 128),
        gaussian_weights=True,
        sigma=1.5,
        use_sample_covariance=False)
    
def calculateEntropy(y):
    entropy = skimage.measure.shannon_entropy(y)
    return entropy

def calculateRuntime(autoencoder,x_test):
    start = time.time()
    decoded_imgs = autoencoder.predict(x_test)
    return time.time() - start

def printMetricsforX(y_true,y_pred,n,autoencoder,x_test):
    psnr = []
    ssim = []
    ssim_wang = []
    entropies = []
    mse = []
    nrmse = []
    for i in range(n):
        psnr.append(calculatePSNR(y_true,y_pred))
        ssim.append(calculateSSIM(y_true,y_pred))
        ssim_wang.append(calculateSSIMWang(y_true,y_pred))
        entropies.append(calculateEntropy(y_true,y_pred))
        mse.append(calculateMSE(y_true,y_pred))
        nrmse.append(calculateNRMSE(y_true,y_pred))

    time = calculateRuntime(autoencoder,x_test)

    print(f"PSNR: {np.mean(psnr)}, SSIM: {np.mean(ssim)}, SSIM_WANG: {np.mean(ssim_wang)}, RMSE: {np.mean(mse)}, NRMSE: {np.mean(nrmse)}, Mean Entropy:{np.mean(entropies)}")
    print(f"Time taken: {time}")


def printMetricsForY(y_true,y_pred):
    psnr = calculatePSNR(y_true,y_pred)
    ssim = calculateSSIM(y_true,y_pred)
    ssim_wang = calculateSSIMWang(y_true,y_pred)
    entropies = calculateEntropy(y_true,y_pred)
    mse = calculateMSE(y_true,y_pred)
    nrmse = calculateNRMSE(y_true,y_pred)

    print(f"PSNR: {psnr}, SSIM: {ssim}, SSIM_WANG: {ssim_wang}, RMSE: {mse}, NRMSE: {nrmse}, Mean Entropy:{entropies}")