import numpy as np
import skimage
import matplotlib.pyplot as plt

def pretty_print(data, n=10, image_dims=(128, 128)):
    plt.figure(figsize=(2 * n, 2))
    for i in range(1, n+1):
        ax = plt.subplot(1, n, i)
        plt.imshow(data[i].reshape(image_dims))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
    
def pretty_print_t(data, title, n=10, image_dims=(128, 128)):
    plt.figure(figsize=(2 * n, 2))
    for i in range(1, n+1):
        ax = plt.subplot(1, n, i)
        plt.imshow(data[i][0].reshape(image_dims))
        plt.gray()
        ax.set_title(title.format(data[i][1]))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
    
def show_results(x_test, y_test, predicted_y, image_dims=(128, 128)):
    psnr = []
    ssim = []
    wang = [] # Z. Wang et al., 2004
    wang_img = []
    for i in range(len(x_test)):
        psnr.append(skimage.measure.compare_psnr(y_test[i], predicted_y[i]))
        ssim.append(skimage.measure.compare_ssim(
            y_test[i].reshape(image_dims), 
            predicted_y[i].reshape(image_dims))
                   )
        wang.append(skimage.measure.compare_ssim(
            y_test[i].reshape(image_dims), 
            predicted_y[i].reshape(image_dims),
            gaussian_weights=True,
            sigma=1.5,
            use_sample_covariance=False
        ))
        wang_img.append((predicted_y[i].reshape(image_dims), skimage.measure.compare_ssim(
            y_test[i].reshape(image_dims), 
            predicted_y[i].reshape(image_dims),
            gaussian_weights=True,
            sigma=1.5,
            use_sample_covariance=False
        )))
    print("PSNR is {:.2f}".format(np.mean(psnr)))
    print("Generic SSIM is {:.2%}".format(np.mean(ssim)))
    print("Wang SSIM is {:.2%}".format(np.mean(wang)))
    pretty_print_t(wang_img, "W SSIM: {:.1%}")
    
def import_dataset(folder_path, num_samples, image_dims):
    x = np.ndarray((num_samples, image_dims[0], image_dims[1]))
    y = np.ndarray((num_samples, image_dims[0], image_dims[1]))
    for i in range(num_samples):
        x[i] = skimage.io.imread("{}/f_fbp/{}.tif".format(folder_path, i))
        y[i] = skimage.io.imread("{}/f_true/{}.tif".format(folder_path, i))

    # Normalize values
    x = x.astype('float32') / 255.
    y = y.astype('float32') / 255.

    # Desired shape for keras is (n_samples, x, y, n_channels)
    x = np.reshape(x, (len(x), image_dims[0], image_dims[1], 1))
    y = np.reshape(y, (len(y), image_dims[0], image_dims[1], 1))
    
    return x, y