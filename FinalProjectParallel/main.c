#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <string.h>

#define MAX_POINTS 1000
#define MAX_LINE 128
#define MAX_LINES_PER_RANK 1000

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

int main(int argc, char* argv[]) {
    int rank, size;
    int N, K, TCount;
    double D;
    PointParams points[MAX_POINTS];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double start_time = MPI_Wtime();

    if (rank == 0) {
        FILE *file = fopen("input.txt", "r");
        if (!file) {
            printf("Error opening input.txt\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
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

        FILE *clear = fopen("output.txt", "w");
        if (clear) fclose(clear);
    }

    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&K, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&D, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&TCount, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(points, N * sizeof(PointParams), MPI_BYTE, 0, MPI_COMM_WORLD);

    int start_t = (TCount + 1) * rank / size;
    int end_t = (TCount + 1) * (rank + 1) / size;

    char lines_buffer[MAX_LINES_PER_RANK][MAX_LINE];
    int line_count = 0;

    for (int i = start_t; i < end_t; i++) {
        double t = 2.0 * i / TCount - 1.0;

        Position positions[MAX_POINTS];

        #pragma omp parallel for
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
                            if (line_count < MAX_LINES_PER_RANK) {
                                snprintf(lines_buffer[line_count++], MAX_LINE,
                                    "Points %d, %d, %d, %d satisfy Proximity Criteria at t = %.4f\n",
                                    positions[i1].id, positions[i2].id,
                                    positions[i3].id, positions[i4].id, t);
                            }
                        }
                    }
                }
            }
        }
    }

    if (rank == 0) {
        FILE *out = fopen("output.txt", "a");
        for (int i = 0; i < line_count; i++) {
            fputs(lines_buffer[i], out);
        }
        for (int src = 1; src < size; src++) {
            int recv_count = 0;
            MPI_Recv(&recv_count, 1, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int i = 0; i < recv_count; i++) {
                char line[MAX_LINE];
                MPI_Recv(line, MAX_LINE, MPI_CHAR, src, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                fputs(line, out);
            }
        }
        fclose(out);
    } else {
        MPI_Send(&line_count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        for (int i = 0; i < line_count; i++) {
            MPI_Send(lines_buffer[i], MAX_LINE, MPI_CHAR, 0, 1, MPI_COMM_WORLD);
        }
    }

    double end_time = MPI_Wtime();
    double local_elapsed = end_time - start_time;
    double max_elapsed;

    MPI_Reduce(&local_elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Total elapsed time (parallel): %.6f seconds\n", max_elapsed);
    }

    MPI_Finalize();
    return 0;
}

