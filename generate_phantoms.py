import os
import time
import astra # https://anaconda.org/astra-toolbox/astra-toolbox
import skimage
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

# Synthetic Phantom Generator

def phantom(n=128, ellipses=None, nE=5, binary=False):
    """
    ellipses : Custom set of ellipses to use.  These should be in 
      the form
        [[a, b, x0, y0, phi],
         [a, b, x0, y0, phi],
         ...]
      where each row defines an ellipse.
      a : Length of the major axis.
      b : Length of the minor axis.
      x0 : Horizontal offset of the centre of the ellipse.
      y0 : Vertical offset of the centre of the ellipse.
      phi : Counterclockwise rotation of the ellipse in degrees,
            measured as the angle between the horizontal axis and 
            the ellipse major axis.
      The image bounding box in the algorithm is [-1, -1], [1, 1], 
      so the values of a, b, x0, y0 should all be specified with
      respect to this box.
    """
    if not ellipses:
        ellipses = []
        for _ in range(nE):
            ellipses.append(
                [
                    (0.25 + 0.25 * np.random.random()),
                    (0.25 + 0.25 * np.random.random()), # Between 0.25 and 0.5
                    (np.random.random() - 0.5), # Between -0.5 and 0.5
                    (np.random.random() - 0.5),
                    (np.random.random() - 0.5),
                ]
            )

    # Blank image
    p = np.zeros((n, n))

    # Create the pixel grid
    ygrid, xgrid = np.mgrid[-1 : 1 : (1j * n), -1 : 1 : (1j * n)]

    for ellip in ellipses:
        a2 = ellip[0] ** 2
        b2 = ellip[1] ** 2
        x0 = ellip[2]
        y0 = ellip[3]
        phi = ellip[4] * np.pi / 180  # Rotation angle in radians

        # Create the offset x and y values for the grid
        x = xgrid - x0
        y = ygrid - y0

        cos_p = np.cos(phi)
        sin_p = np.sin(phi)

        # Find the pixels within the ellipse
        locs = (
            ((x * cos_p + y * sin_p) ** 2) / a2 + ((y * cos_p - x * sin_p) ** 2) / b2
        ) <= 1

        # Add the ellipse intensity to those pixels
        if binary:
            p[locs] = 255
        else:
            p[locs] = 255 * (0.5 + 0.5 * np.random.random()) # Between 128 and 255
    return p


def create_sino_recon_pairs(f_true, noise_func=lambda x: x, recon_alg="FBP"):

    # Create volume geometries
    v, h = f_true.shape
    vol_geom = astra.create_vol_geom(v, h)

    # Create projector geometries
    det_count = f_true.shape[0]
    angles = np.linspace(0, np.pi, 180, endpoint=False)
    proj_geom = astra.create_proj_geom("parallel", 1.0, det_count, angles)

    # Create projector
    projector_id = astra.create_projector("strip", proj_geom, vol_geom)

    # Radon transform (generate sinogram)
    sinogram_id, sinogram = astra.create_sino(f_true, projector_id)

    # Add noise in the sinogram domain
    sino_geom = astra.data2d.get_geometry(sinogram_id)
    sinogram = noise_func(sinogram)
    sinogram_id = astra.data2d.create("-sino", sino_geom, data=sinogram)

    # Create a data object for the reconstruction
    rec_id = astra.data2d.create("-vol", vol_geom)

    # Set up the parameters for a reconstruction via back-projection
    cfg = astra.astra_dict(recon_alg)
    cfg["ReconstructionDataId"] = rec_id
    cfg["ProjectionDataId"] = sinogram_id
    cfg["ProjectorId"] = projector_id

    # Create the algorithm object from the configuration structure
    alg_id = astra.algorithm.create(cfg)

    # Run back-projection and get the reconstruction
    astra.algorithm.run(alg_id)
    f_rec = astra.data2d.get(rec_id)
    return sinogram, f_rec


def noisy(s):
    noise = np.random.normal(0, 255, s.shape)

    return s + noise


def in_painting(s):
    res = s
    start = int(2 * s.shape[0] / 5)
    end = int(3 * s.shape[0] / 5)
    res[start:end, :] = 0
    return res


def undersampling(s):
    res = s
    fraction = 20
    for i in range(fraction):
        start = int((i + 0.25) * s.shape[0] / fraction)
        end = int((i + 0.75) * s.shape[0] / fraction)
        res[start:end, :] = 0
    return res


def generate_one(noise_func=noisy, binary=False):
    f_true = phantom(binary=binary)
    s, f_fbp = create_sino_recon_pairs(f_true, noise_func=noise_func, recon_alg="FBP")
    s, f_bp = create_sino_recon_pairs(f_true, noise_func=noise_func, recon_alg="BP")
    return f_true, s, f_fbp, f_bp


def rescale(data):
    return (255.0 / data.max() * (data - data.min())).astype(np.uint8)


def generate(n, folder_path, noise_type, binary):
    start = time.time()
    os.mkdir("{}/f_true".format(folder_path))
    os.mkdir("{}/sino".format(folder_path))
    os.mkdir("{}/f_fbp".format(folder_path))
    os.mkdir("{}/f_bp".format(folder_path))

    noise_func = noisy
    if noise_type == "inpainting":
        noise_func = in_painting
    elif noise_type == "undersampling":
        noise_func = undersampling

    for i in range(n):
        f_true, s, f_fbp, f_bp = generate_one(noise_func=noise_func, binary=binary)
        Image.fromarray(rescale(f_true)).save("{}/f_true/{}.tif".format(folder_path, i))
        Image.fromarray(rescale(s)).save("{}/sino/{}.tif".format(folder_path, i))
        Image.fromarray(rescale(f_fbp)).save("{}/f_fbp/{}.tif".format(folder_path, i))
        Image.fromarray(rescale(f_bp)).save("{}/f_bp/{}.tif".format(folder_path, i))
    print("Generated {} in {:.1f} seconds!".format(n, time.time() - start))


def main():
    try:
        num_samples = int(input("How many samples? [100] "))
    except ValueError:
        num_samples = 100

    folder_path = str(input("What folder would you like it in? [./data/pairs] "))
    if folder_path == "":
        folder_path = "data/pairs"
    try:
        os.mkdir(folder_path)
        print("Starting...")
        
        noise_type = str(input("Noise type? (gaussian/inpainting/[undersampling]) "))
        if noise_type == "":
            noise_type = "undersampling"
        
        ground_truth_type = str(input("Construct Ground truth from binary image? (y/[N]) "))
        binary = False
        if ground_truth_type == "y":
            binary = True
        
        generate(num_samples, folder_path, noise_type=noise_type, binary=binary)
        
    except:
        print("Folder already exists: Quitting.")


if __name__ == "__main__":
    main()

