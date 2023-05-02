import skimage
from helpers import save_loss
from functions import create_mesh, compute_loss, step_mesh, find_surface
import numpy as np
from poisson import poisson
from tqdm import tqdm


def iteration(mesh, gray, prefix):
    height, width = gray.shape
    loss = compute_loss(mesh, gray)
    save_loss(loss, prefix)

    phi = np.zeros((width, height), np.double)
    loss = loss.astype(np.double)

    for i in range(10000):
        max_update = poisson(phi, loss, width, height)

        if i % 500 == 0:
            print(max_update)
        
        if max_update < 0.0001:
            print("Converged on step", i)
            break
    step_mesh(mesh, phi)


def main():
    img = skimage.io.imread("images/cat_posing.jpg")
    gray = skimage.color.rgb2gray(img)
    height, width = gray.shape[:2]

    mesh_sum = width * height
    gray_sum = gray.sum()
    boost_ratio = mesh_sum / gray_sum

    gray *= boost_ratio
    mesh = create_mesh(width + 1, height + 1)

    for i in tqdm(range(4)):
        iteration(mesh, gray, f"it{i}")
    
    artifactSize = 0.1  # meters
    focalLength = 0.2 # meters
    h, metersPerPixel = find_surface(mesh, gray, focalLength, artifactSize)


if __name__ == "__main__":
    main()
