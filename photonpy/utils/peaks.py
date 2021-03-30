import numpy as np
from skimage.feature import peak_local_max


def get_peak_positions(image, min_distance=5, threshold_rel=0.5, plot=False):
    positions = peak_local_max(
        image, min_distance=min_distance, threshold_rel=threshold_rel, indices=True, threshold_abs=np.min(image) * 5
    )

    if plot:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.imshow(image)
        plt.plot(positions[:, 1], positions[:, 0], "rx")
        plt.show()

    return positions


def get_rois(image, positions, roi_size=15, plot=False):
    imgs = []
    for pos in positions:
        x = slice(int(pos[1] - roi_size / 2), int(pos[1] + roi_size / 2))
        y = slice(int(pos[0] - roi_size / 2), int(pos[0] + roi_size / 2))
        if len(image.shape) == 2:
            roi = image[y, x]
            if roi.shape == (roi_size, roi_size):
                imgs.append(roi)
        else:
            roi = image[:, y, x]
            if roi.shape[1:] == (roi_size, roi_size):
                imgs.append(roi)


    if plot:
        import matplotlib.pyplot as plt

        plt.figure()
        for i, img in enumerate(imgs):
            plt.subplot(len(imgs), 1, i + 1)
            plt.imshow(img)
        plt.show()

    return np.asarray(imgs)


def get_rois_corner(image, positions, roi_size=15):
    imgs = []
    for pos in positions:
        x = slice(int(pos[0]), int(pos[0] + roi_size))
        y = slice(int(pos[1]), int(pos[1] + roi_size))
        if len(image.shape) == 2:
            roi = image[y, x]
            if roi.shape == (roi_size, roi_size):
                imgs.append(roi)
        else:
            roi = image[:, y, x]
            if roi.shape[1:] == (roi_size, roi_size):
                imgs.append(roi)
    return np.asarray(imgs)
