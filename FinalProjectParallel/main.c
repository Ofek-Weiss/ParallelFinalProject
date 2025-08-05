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

// Optimized version that stops early
int count_neighbors_optimized(Position* pos, int n, int point_idx, double d, int min_neighbors) {
    int count = 0;
    // No OpenMP here - already called from parallel region
    for (int j = 0; j < n; j++) {
        if (j != point_idx) {
            double dx = pos[point_idx].x - pos[j].x;
            double dy = pos[point_idx].y - pos[j].y;
            double dist = sqrt(dx * dx + dy * dy);
            if (dist < d) {
                count++;
                if (count >= min_neighbors) {  // Stop early when we have enough neighbors
                    return count;
                }
            }
        }
    }
    return count;
}

int find_trio(Position* pos, int n, int k, double d, int* trio) {
    int found = 0;
    int count = 0;
    
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; i++) {
        // Check criteria directly - no need for separate function
        int neighbors = count_neighbors_optimized(pos, n, i, d, k);
        if (neighbors >= k) {
            #pragma omp critical
            {
                if (count < 3) {
                    trio[count] = i;
                    count++;
                    if (count == 3) {
                        found = 1;
                    }
                }
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

    double start_time = MPI_Wtime();
    
    // Set OpenMP threads dynamically based on CPU
    int max_threads = omp_get_num_procs() / size;
    if (max_threads < 1) max_threads = 1;
    omp_set_num_threads(max_threads);
    if (rank == 0) {
        printf("MPI size: %d processes, each with %d OpenMP threads (total %d cores available)\n", size, max_threads, omp_get_num_procs());
    }
    
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
    
    // Divide work among processes - ensure we process tcount + 1 values
    int total_steps = tcount + 1;
    int work_per_process = total_steps / size;
    int extra_work = total_steps % size;
    int start_t = rank * work_per_process + (rank < extra_work ? rank : extra_work);
    int end_t = start_t + work_per_process + (rank < extra_work ? 1 : 0);
    
    Position pos[MAX_POINTS];
    char results[1000][200];
    int result_count = 0;
    int found_count = 0;
    
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
        int trio[3];
        if (find_trio(pos, n, k, d, trio)) {
            found_count++;
            if (result_count < 1000) {
                sprintf(results[result_count], 
                    "Points %d, %d, %d satisfy criteria at t = %.4f\n",
                    trio[0], trio[1], trio[2], t);
                result_count++;
            }
        }
    }
    
    // Gather results to rank 0
    int all_found = 0;
    MPI_Reduce(&found_count, &all_found, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    
    // Gather result counts from all processes
    int* result_counts = NULL;
    if (rank == 0) {
        result_counts = (int*)malloc(size * sizeof(int));
    }
    MPI_Gather(&result_count, 1, MPI_INT, result_counts, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Debug: print result counts
    if (rank == 0) {
        printf("Debug: Results per process: ");
        for (int i = 0; i < size; i++) {
            printf("Process %d: %d, ", i, result_counts[i]);
        }
        printf("\n");
    }
    
    // Gather all result strings from all processes using simple MPI_Gather
    char* all_results = NULL;
    
    if (rank == 0) {
        all_results = (char*)malloc(size * 1000 * 200 * sizeof(char)); // Max 1000 results per process
    }
    
    // Gather all result strings - each process sends 1000 * 200 bytes
    MPI_Gather(results, 1000 * 200, MPI_CHAR, 
               all_results, 1000 * 200, MPI_CHAR, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Total trios found: %d\n", all_found);
        
        // Write results to file
        FILE* outfile = fopen("output.txt", "w");
        if (outfile) {
            // Write results from all processes
            for (int i = 0; i < size; i++) {
                char* process_results = all_results + (i * 1000 * 200);
                printf("Debug: Writing %d results from process %d\n", result_counts[i], i);
                // Write the actual results from each process
                for (int j = 0; j < result_counts[i]; j++) {
                    char* result_str = process_results + (j * 200);
                    fprintf(outfile, "%s", result_str);
                }
            }
            
            // Write message if no trios found
            if (all_found == 0) {
                fprintf(outfile, "There were no 3 points found for any t.\n");
            }
            
            fclose(outfile);
        }
        
        // Clean up
        free(result_counts);
        free(all_results);
    }
    
    double end_time = MPI_Wtime(); // Measure time after ALL operations
    
    if (rank == 0) {
        printf("Total elapsed time: %.6f seconds\n", end_time - start_time);
    }
    
    MPI_Finalize();
    return 0;
}

