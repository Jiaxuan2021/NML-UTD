import numpy as np
import matplotlib.pyplot as plt

def Normalize_(input_spectra):
    """
    input_spectra : row * col
    """
    r, c = input_spectra.shape
    input_spectra = input_spectra.flatten()
    min = np.min(input_spectra)  
    max = np.max(input_spectra)
    result = (input_spectra - min) / (max - min)
    return result.reshape(r, c)

def Threshold_(result, threshold=0.1):
    """
    Threshold binarization of confidence
    threshold * n
    """
    r, c = result.shape
    result = result.flatten()
    n = result.shape[0]
    result_sort = sorted(result, reverse=True)
    threshold_ = result_sort[int(threshold * n)]
    # print(threshold_)
    threshold_result = np.zeros_like(result)
    threshold_result[np.where(result >= threshold_)] = 255
    return threshold_result.reshape(r, c)

def Threshold_2(result, threshold=0.1):
    """
    Threshold binarization of confidence
    threshold * n
    save the initial value, discard the value less than threshold
    """
    r, c = result.shape
    result = result.flatten()
    n = result.shape[0]
    result_sort = sorted(result, reverse=True)
    threshold_ = result_sort[int(threshold * n)]
    # print(threshold_)
    threshold_result = result
    threshold_result[np.where(result < threshold_)] = 0
    return threshold_result.reshape(r, c)

def Gamma_(result, gamma=0.5):
    """
    Gamma correction
    """
    r, c = result.shape
    result = result.flatten()
    result = np.power(result, gamma)
    return result.reshape(r, c)

def threshold(result, threshold, gamma=None):
    """
    result : row * col
    """
    result = Normalize_(result)
    # result = Threshold_(result, threshold)
    result = Threshold_(result, threshold)
    if gamma == None:
        return result
    else:
        result = Gamma_(result, gamma)
    return result

if __name__ == '__main__':
    result = np.load(fr'../result_detection/River_scene1/result_45.npy')
    result = threshold(result, 0.07)
    plt.imshow(result)
    plt.axis('off')
    plt.savefig('threshold.png', bbox_inches='tight', pad_inches=0)
    plt.close()