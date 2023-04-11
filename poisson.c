#include <stdlib.h>
#include <stdio.h>
#include <math.h>

double sum(double *arr, int size) {
    double s = 0.0;
    for (int i = 0; i < size; i++) {
        s += arr[i];
    }
    return s;
}

int get_adjacent_coords(int *res, int x, int y, int width, int height) {
    int candidates[8] = {x - 1, y, x + 1, y, x, y - 1, x, y + 1};
    int size = 0;
    for (int i = 0; i < 8; i += 2) {
        int ix = candidates[i];
        int iy = candidates[i+1];
        if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
            res[size++] = ix + iy * width;
        }
    }
    return size;
}

int get_adjacent_points(double *res, double *points, int x, int y, int width, int height) {
    int *adjacents = (int*)malloc(sizeof(int) * 4);
    int adjacents_size = get_adjacent_coords(adjacents, x, y, width, height);

    for (int i = 0; i < adjacents_size; i++) {
        res[i] = points[adjacents[i]];
    }
    free(adjacents);
    return adjacents_size;
}

double poisson(double *f, double *h, int width, int height) {
    double omega = 1.99;
    double max_update = 0.0;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            double *points = (double*)malloc(sizeof(double) * 4);
            double adjacents_size = (double)get_adjacent_points(points, f, x, y, width, height);
            double delta = omega / adjacents_size * (sum(points, adjacents_size) - adjacents_size * f[x + y * width] - h[x + y * width]);
            if (fabs(delta) > max_update) {
                max_update = fabs(delta);
            }
            f[x + y * width] += delta;
            free(points);
        }
    }
    return max_update;
}

int main() {
    return 0;
}