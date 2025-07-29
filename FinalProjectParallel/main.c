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

// Count how many points are within distance D of a given point
int count_proximity_neighbors(Position* positions, int N, int point_idx, double D) {
    int count = 0;
    for (int j = 0; j < N; j++) {
        if (j != point_idx) {
            double dist = hypot(positions[point_idx].x - positions[j].x, 
                              positions[point_idx].y - positions[j].y);
            if (dist < D) count++;
        }
    }
    return count;
}

// Check if a point satisfies Proximity Criteria (has at least K neighbors within distance D)
int satisfies_proximity_criteria(Position* positions, int N, int point_idx, int K, double D) {
    return count_proximity_neighbors(positions, N, point_idx, D) >= K;
}

// Find 4 points that all satisfy Proximity Criteria
int find_proximity_quartet(Position* positions, int N, int K, double D, int* quartet) {
    // Try multiple different starting points to get variety
    int start_points[] = {0, N/4, N/2, 3*N/4};
    
    for (int start_idx = 0; start_idx < 4; start_idx++) {
        int start = start_points[start_idx];
        
        for (int i1 = start; i1 < N - 3; i1++) {
            if (!satisfies_proximity_criteria(positions, N, i1, K, D)) continue;
            
            for (int i2 = i1 + 1; i2 < N - 2; i2++) {
                if (!satisfies_proximity_criteria(positions, N, i2, K, D)) continue;
                
                for (int i3 = i2 + 1; i3 < N - 1; i3++) {
                    if (!satisfies_proximity_criteria(positions, N, i3, K, D)) continue;
                    
                    for (int i4 = i3 + 1; i4 < N; i4++) {
                        if (satisfies_proximity_criteria(positions, N, i4, K, D)) {
                            quartet[0] = i1;
                            quartet[1] = i2;
                            quartet[2] = i3;
                            quartet[3] = i4;
                            return 1; // Found a quartet
                        }
                    }
                }
            }
        }
    }
    
    return 0; // No quartet found
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

    // Better load balancing - each process gets chunks of work
    int total_work = TCount + 1;
    int work_per_process = total_work / size;
    int extra_work = total_work % size;
    
    int start_t = rank * work_per_process + (rank < extra_work ? rank : extra_work);
    int end_t = start_t + work_per_process + (rank < extra_work ? 1 : 0);

    char lines_buffer[MAX_LINES_PER_RANK][MAX_LINE];
    int line_count = 0;
    int total_found = 0;

    for (int i = start_t; i < end_t; i++) {
        double t = 2.0 * i / TCount - 1.0;

        Position positions[MAX_POINTS];
        
        // Remove OpenMP from small loop - it's overhead for small N
        for (int j = 0; j < N; j++) {
            double x, y;
            compute_position(points[j], t, &x, &y);
            positions[j].x = x;
            positions[j].y = y;
            positions[j].id = points[j].id;
        }

        int quartet[4];
        if (find_proximity_quartet(positions, N, K, D, quartet)) {
            total_found++;
            if (line_count < MAX_LINES_PER_RANK) {
                snprintf(lines_buffer[line_count++], MAX_LINE,
                    "Points %d, %d, %d, %d satisfy Proximity Criteria at t = %.4f\n",
                    positions[quartet[0]].id, positions[quartet[1]].id,
                    positions[quartet[2]].id, positions[quartet[3]].id, t);
            }
        }
    }

    // Gather total found count from all processes
    int global_total_found;
    MPI_Reduce(&total_found, &global_total_found, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        FILE *out = fopen("output.txt", "a");
        
        // Write results from rank 0
        for (int i = 0; i < line_count; i++) {
            fputs(lines_buffer[i], out);
        }
        
        // Receive and write results from other ranks
        for (int src = 1; src < size; src++) {
            int recv_count = 0;
            MPI_Recv(&recv_count, 1, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int i = 0; i < recv_count; i++) {
                char line[MAX_LINE];
                MPI_Recv(line, MAX_LINE, MPI_CHAR, src, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                fputs(line, out);
            }
        }
        
        // Write fallback message if no points found
        if (global_total_found == 0) {
            fprintf(out, "There were no 4 points found for any t.\n");
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
        printf("Total elapsed time (MPI+OpenMP): %.6f seconds\n", max_elapsed);
    }

    MPI_Finalize();
    return 0;
}

