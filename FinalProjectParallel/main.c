#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>

#define MAX_POINTS 2000

typedef struct {
    int id;
    double x1, x2, a, b;
} Point;

typedef struct {
    int id;
    double x, y;
} Position;

void compute_position(Point p, double t, double *x, double *y) {
    double temp_x = ((p.x2 + p.x1) / 2.0) * cos(t * M_PI / 2.0) + (p.x2 - p.x1) / 2.0;
    *x = temp_x;
    *y = p.a * temp_x + p.b;
}

int count_neighbors(Position* pos, int n, int point_idx, double d) {
    int count = 0;
    #pragma omp parallel for reduction(+:count)
    for (int j = 0; j < n; j++) {
        if (j != point_idx) {
            double dx = pos[point_idx].x - pos[j].x;
            double dy = pos[point_idx].y - pos[j].y;
            double dist = sqrt(dx * dx + dy * dy);
            if (dist < d) count++;
        }
    }
    return count;
}

int check_criteria(Position* pos, int n, int point_idx, int k, double d) {
    int neighbors = count_neighbors(pos, n, point_idx, d);
    return (neighbors >= k);
}

int find_quartet(Position* pos, int n, int k, double d, int* quartet) {
    int found = 0;
    int count = 0;
    
    for (int i = 0; i < n && !found; i++) {
        if (check_criteria(pos, n, i, k, d)) {
            quartet[count] = i;
            count++;
            if (count == 4) {
                found = 1;
                break;
            }
        }
    }
    return found;
}

int main(int argc, char* argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Set OpenMP threads
    omp_set_num_threads(2);
    
    if (size < 2) {
        if (rank == 0) {
            printf("Need at least 2 processes\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    int n, k, tcount;
    double d;
    Point points[MAX_POINTS];
    
    // Read input on rank 0
    if (rank == 0) {
        FILE* file = fopen("input.txt", "r");
        if (!file) {
            printf("Cannot open input.txt\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        fscanf(file, "%d %d %lf %d", &n, &k, &d, &tcount);
        for (int i = 0; i < n; i++) {
            fscanf(file, "%d %lf %lf %lf %lf", &points[i].id, &points[i].x1, 
                   &points[i].x2, &points[i].a, &points[i].b);
        }
        fclose(file);
    }
    
    // Broadcast parameters
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&k, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&tcount, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(points, n * sizeof(Point), MPI_BYTE, 0, MPI_COMM_WORLD);
    
    // Divide work among processes
    int work_per_process = tcount / size;
    int extra_work = tcount % size;
    int start_t = rank * work_per_process + (rank < extra_work ? rank : extra_work);
    int end_t = start_t + work_per_process + (rank < extra_work ? 1 : 0);
    
    Position pos[MAX_POINTS];
    char results[1000][200];
    int result_count = 0;
    int found_count = 0;
    
    double start_time = MPI_Wtime();
    
    // Process assigned time steps
    for (int t_idx = start_t; t_idx < end_t; t_idx++) {
        double t = 2.0 * t_idx / tcount - 1.0;
        
        // Compute positions for all points
        #pragma omp parallel for
        for (int j = 0; j < n; j++) {
            double x, y;
            compute_position(points[j], t, &x, &y);
            pos[j].x = x;
            pos[j].y = y;
            pos[j].id = points[j].id;
        }
        
        // Find quartet
        int quartet[4];
        if (find_quartet(pos, n, k, d, quartet)) {
            found_count++;
            if (result_count < 1000) {
                sprintf(results[result_count], 
                    "Points %d, %d, %d, %d satisfy criteria at t = %.4f\n",
                    quartet[0], quartet[1], quartet[2], quartet[3], t);
                result_count++;
            }
        }
    }
    
    double end_time = MPI_Wtime();
    
    // Gather results to rank 0
    int all_found = 0;
    MPI_Reduce(&found_count, &all_found, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Total elapsed time: %.6f seconds\n", end_time - start_time);
        printf("Total quartets found: %d\n", all_found);
        
        // Write results to file
        FILE* outfile = fopen("output.txt", "w");
        if (outfile) {
            for (int i = 0; i < result_count; i++) {
                fprintf(outfile, "%s", results[i]);
            }
            fclose(outfile);
        }
    }
    
    MPI_Finalize();
    return 0;
}

