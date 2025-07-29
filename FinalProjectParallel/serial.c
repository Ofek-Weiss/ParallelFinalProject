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

    int total_found = 0;

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

        int quartet[4];
        if (find_proximity_quartet(positions, N, K, D, quartet)) {
            total_found++;
            FILE *out = fopen("output.txt", "a");
            if (out) {
                fprintf(out,
                    "Points %d, %d, %d, %d satisfy Proximity Criteria at t = %.4f\n",
                    positions[quartet[0]].id, positions[quartet[1]].id,
                    positions[quartet[2]].id, positions[quartet[3]].id, t);
                fclose(out);
            }
        }
    }

    // Write fallback message if no points found
    if (total_found == 0) {
        FILE *out = fopen("output.txt", "a");
        if (out) {
            fprintf(out, "There were no 4 points found for any t.\n");
            fclose(out);
        }
    }

    clock_t end = clock();  // End timing
    double elapsed_secs = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Total elapsed time (serial): %.6f seconds\n", elapsed_secs);

    return 0;
}

