import os
import numpy as np
import tomopy
from PIL import Image

add_noise = True
n_samples = 2000
allow_existing_folder = True
initialOffset = 0
folder_path = "out_noisy_pairs"


def bnw(image):
    for i in range(len(image)):
        for j in range(len(image[i])):
            if image[i][j] < 1:
                image[i][j] = 0
            else:
                image[i][j] = 1


def noisy(noise_typ, image):
    if noise_typ == "gauss":
        row, col, ch = image.shape
        mean = 0
        var = 0.1
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss
        return noisy
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ == "speckle":
        row, col, ch = image.shape
        gauss = np.random.randn(row, col, ch)
        gauss = gauss.reshape(row, col, ch)
        noisy = image + image * gauss
        return noisy
    else:
        print("Unknown Noise")


def crop_center(img, cropx, cropy):
    y, x = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty : starty + cropy, startx : startx + cropx]


def ran():
    return -0.5 + 1 * np.random.random()


def phantom(n=128, ellipses=None, nE=5):
    """
    ellipses : Custom set of ellipses to use.  These should be in 
      the form
        [[I, a, b, x0, y0, phi],
         [I, a, b, x0, y0, phi],
         ...]
      where each row defines an ellipse.
      I : Additive intensity of the ellipse.
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
                    ran(),
                    ran(),
                    ran(),
                    ran(),
                    ran(),  # np.random.random(),
                    # np.random.random(),
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
        p[locs] = 1
    return p


def rescale(data):
    return (255.0 / data.max() * (data - data.min())).astype(np.uint8)


def main():
    offset = initialOffset
    keepGoing = "Y"
    while not keepGoing == "n":
        print("Generating samples {} to {}".format(offset, n_samples + offset))
        for i in range(n_samples):
            iteration = i + offset
            clean = np.array([phantom()])

            if add_noise:
                obj = noisy("gauss", clean)
            else:
                obj = clean

            Image.fromarray(rescale(clean[0])).save(
                "{}/{}_actual.png".format(folder_path, iteration)
            )
            Image.fromarray(rescale(obj[0])).save(
                "{}/{}_noisy.png".format(folder_path, iteration)
            )

            ang = tomopy.angles(180)
            sim = tomopy.project(obj, ang)

            algs = ["fbp", "gridrec"]
            for alg in algs:
                rec = tomopy.recon(sim, ang, algorithm=alg, ncore=4)
                crop_rec = crop_center(rec[0], 128, 128)
                Image.fromarray(rescale(crop_rec)).save(
                    "{}/{}_recon_{}.png".format(folder_path, iteration, alg)
                )
        offset += n_samples
        keepGoing = input("Create {} more? ([Y]/n) ".format(n_samples))


if __name__ == "__main__":
    go = False
    try:
        os.mkdir(folder_path)
        go = True
        print("Starting...")
    except:
        if allow_existing_folder:
            go = True
        else:
            print("Folder already exists: Quitting.")

    if go:
        main()
