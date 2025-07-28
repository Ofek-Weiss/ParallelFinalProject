#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define MAX_POINTS 1000

typedef struct {
    int id;
    double x1, x2, a, b;
} PointParams;

typedef struct {
    int id;
    double x, y;
} Position;

void compute_position(PointParams p, double t, double *x, double *y) {
    *x = ((p.x2 + p.x1) / 2.0) * cos(t * M_PI / 2.0) + (p.x2 - p.x1) / 2.0;
    *y = p.a * (*x) + p.b;
}

int main() {
    int N, K, TCount;
    double D;
    PointParams points[MAX_POINTS];

    // Read input from file
    FILE *file = fopen("input.txt", "r");
    if (!file) {
        printf("Error opening input.txt\n");
        return 1;
    }

    fscanf(file, "%d %d %lf %d", &N, &K, &D, &TCount);
    for (int i = 0; i < N; i++) {
        fscanf(file, "%d %lf %lf %lf %lf",
               &points[i].id,
               &points[i].x1,
               &points[i].x2,
               &points[i].a,
               &points[i].b);
    }
    fclose(file);

    // Clear output file
    FILE *clear = fopen("output.txt", "w");
    if (clear) fclose(clear);

    clock_t start = clock();  // Start timing

    for (int i = 0; i <= TCount; i++) {
        double t = 2.0 * i / TCount - 1.0;

        Position positions[MAX_POINTS];
        for (int j = 0; j < N; j++) {
            double x, y;
            compute_position(points[j], t, &x, &y);
            positions[j].x = x;
            positions[j].y = y;
            positions[j].id = points[j].id;
        }

        int found = 0;

        for (int i1 = 0; i1 < N && !found; i1++) {
            for (int i2 = i1 + 1; i2 < N && !found; i2++) {
                for (int i3 = i2 + 1; i3 < N && !found; i3++) {
                    for (int i4 = i3 + 1; i4 < N && !found; i4++) {
                        double d12 = hypot(positions[i1].x - positions[i2].x, positions[i1].y - positions[i2].y);
                        double d13 = hypot(positions[i1].x - positions[i3].x, positions[i1].y - positions[i3].y);
                        double d14 = hypot(positions[i1].x - positions[i4].x, positions[i1].y - positions[i4].y);
                        double d23 = hypot(positions[i2].x - positions[i3].x, positions[i2].y - positions[i3].y);
                        double d24 = hypot(positions[i2].x - positions[i4].x, positions[i2].y - positions[i4].y);
                        double d34 = hypot(positions[i3].x - positions[i4].x, positions[i3].y - positions[i4].y);

                        if (d12 < D && d13 < D && d14 < D &&
                            d23 < D && d24 < D && d34 < D) {
                            found = 1;

                            FILE *out = fopen("output.txt", "a");
                            if (out) {
                                fprintf(out,
                                    "Points %d, %d, %d, %d satisfy Proximity Criteria at t = %.4f\n",
                                    positions[i1].id, positions[i2].id,
                                    positions[i3].id, positions[i4].id, t);
                                fclose(out);
                            }

                            break;  // Break i4
                        }
                    }
                    if (found) break;  // Break i3
                }
                if (found) break;  // Break i2
            }
            if (found) break;  // Break i1
        }
    }

    clock_t end = clock();  // End timing
    double elapsed_secs = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Total elapsed time (serial): %.6f seconds\n", elapsed_secs);

    return 0;
}

