import argparse
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


def main(image="images/cat_posing.jpg", output="result.obj"):
    img = skimage.io.imread(image)
    if img.ndim == 3 and img.shape[2] == 4:
        img = skimage.color.rgba2rgb(img)
    gray = skimage.color.rgb2gray(img)
    height, width = gray.shape[:2]

    mesh_sum = width * height
    gray_sum = gray.sum()
    boost_ratio = mesh_sum / gray_sum

    gray *= boost_ratio
    mesh = create_mesh(width + 1, height + 1)

    for i in tqdm(range(4)):
        iteration(mesh, gray, f"it{i}")
    
    artifact_size = 0.1  # meters (100 mm square tile)
    focal_length = 0.2 # meters
    h, meters_per_pixel = find_surface(mesh, gray, focal_length, artifact_size)

    set_heights(mesh, h, 1, 1)

    # Size the solid backing to fit the physical stock instead of a fixed offset.
    # Nominal 1/4" acrylic, measured at 5.6 mm; leave a margin so the cutter never reaches the spoilboard.
    stock_thickness = 0.0056  # meters (measured)
    safety_margin = 0.0003     # 0.3 mm
    target_thickness = stock_thickness - safety_margin

    scalez = artifact_size / 512  # meters per mesh z-unit (must match save_obj)
    top_z_max = max(p.z for row in mesh for p in row)  # highest relief point, in units
    offset = target_thickness / scalez - top_z_max  # places the flat back so total thickness == target

    solid_mesh = solidify(mesh, offset=offset)
    save_obj(solid_mesh, output, 1 / 512 * artifact_size, 1 / 512 * artifact_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a caustic lens .obj from an image.")
    parser.add_argument(
        "image",
        nargs="?",
        default="images/cat_posing.jpg",
        help="Path or URL of the source image (default: images/cat_posing.jpg)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="result.obj",
        help="Path for the output .obj file (default: result.obj)",
    )
    args = parser.parse_args()
    main(args.image, args.output)
