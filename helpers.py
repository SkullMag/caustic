import matplotlib.pyplot as plt
import numpy as np
import cv2


def plot_mesh(points):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    for row in points:
        for point in row:
            ax.scatter3D(point.x, point.y, point.z)

    # for triangle in mesh.triangles:
    #     p1 = mesh.points[triangle.a]
    #     p2 = mesh.points[triangle.b]
    #     p3 = mesh.points[triangle.c]
    #     ax.plot3D([p1.x, p2.x, p3.x, p1.x], [p1.y, p2.y, p3.y, p1.y], [p1.z, p2.z, p3.z, p1.z])

    plt.show()
    plt.waitforbuttonpress()


def save_loss(loss: np.ndarray[float], prefix=""):
    h, w = loss.shape[:2]
    result = np.zeros((w, h, 3), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            color = [0, 0, 0] # BGR
            val = loss[y, x]
            if val > 0:
                color[0] = val * 255
            if val < 0:
                color[2] = -val * 255
            result[y, x] = color
    cv2.imwrite(f"images/{prefix}_loss.jpg", result)


def save_obj(mesh, filename: str, scale: float = 1.0, scalez: float = 1.0):
    node_list, _, triangles, width, height = mesh
    with open(filename, "a") as f:
        for i in node_list:
            f.write(f"v {i.x * scale} {i.y * scale} {i.z * scalez}\n")
        
        for i in triangles:
            try:
                f.write(f"f {i.index1} {i.index2} {i.index3}\n")
            except:
                continue
        
        f.write(f"dims {width} {height}")