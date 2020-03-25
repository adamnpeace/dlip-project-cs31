# Applications of Deep Learning for Ill-Posed Inverse Problems Within Optical Tomography 

*Research Conducted for UCL CMIC under Prof Simon Arridge*

Increasingly in medical imaging has emerged an issue surrounding the reconstruction of noisy images from raw scan data. Where the forward problem is the generation of raw measurement data from a ground truth image, the inverse problem is the reconstruction of those images from that data.

In most cases with medical imaging, classical mathematical transforms work well for recovering images from clean measurement data. Unfortunately, this causes an increase of the exposure of patients to radioactive substances and makes medical imaging prohibitively expensive for developing countries.

In our project we aim to show that, by using a cascade of deep neural networks, it is possible to reconstruct usable images from noisy and undersampled scan data to enable the introduction of cheaper and safer scan technology.

## To Use:

(Installation of Astra toolbox requires `conda`)

First ensure the following are installed:

```
numpy
pandas
scikit-image
scikit-learn
matplotlib
tensorflow
keras

astra-toolbox
pillow
```

For using the pre-trained models:

- Run any of the `AE-*.ipynb` notebooks

For training a new model:

- Run `generate_phantoms.py` and select desired phantom parameters

- Modify the `model_file` and `folder_path` and uncomment the `retrain` line in any `AE-*.ipynb` notebook

- Run the notebook
