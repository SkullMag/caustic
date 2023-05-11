import skimage
from helpers import save_loss, save_obj
from functions import create_mesh, compute_loss, step_mesh, find_surface, solidify, set_heights
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
    
    artifact_size = 0.1  # meters
    focal_length = 0.2 # meters
    h, meters_per_pixel = find_surface(mesh, gray, focal_length, artifact_size)

    set_heights(mesh, h, 1, 1)

    solid_mesh = solidify(mesh)
    save_obj(solid_mesh, "result.obj", 1 / 512 * artifact_size, 1 / 512 * artifact_size)


if __name__ == "__main__":
    main()
