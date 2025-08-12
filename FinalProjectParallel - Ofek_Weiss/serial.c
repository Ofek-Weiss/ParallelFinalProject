/* Implementation: Sequential (Serial) Version for Comparison */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define MAX_POINTS 2000


typedef struct {
    int id;        // Unique identifier for the point
    double x1, x2; // Parameters for x-coordinate calculation
    double a, b;   // Parameters for y-coordinate calculation (y = a*x + b)
} Point;


typedef struct {
    int id;        // Point identifier
    double x, y;   // Computed coordinates for current t value
} Position;

/*
 * Computes the (x,y) coordinates for a point given parameter t
 * 
 * Formula: x = ((x2 + x1) / 2) * cos(t*π/2) + (x2 - x1) / 2
 *          y = a*x + b
 */
void compute_position(Point p, double t, double *x, double *y) {
    double temp_x = ((p.x2 + p.x1) / 2.0) * cos(t * M_PI / 2.0) + (p.x2 - p.x1) / 2.0;
    *x = temp_x;
    *y = p.a * temp_x + p.b;
}


int count_neighbors(Position* pos, int n, int point_idx, double d) { // Counts the number of neighbors within distance D from a given point
    int count = 0;
    for (int j = 0; j < n; j++) {
        if (j != point_idx) {
            double dx = pos[point_idx].x - pos[j].x;
            double dy = pos[point_idx].y - pos[j].y;
            double dist = sqrt(dx * dx + dy * dy);
            if (dist < d) {
                count++;
                if (count >= 3) {  // Stop early when we have enough neighbors
                    return count;
                }
            }
        }
    }
    return count;
}


int check_criteria(Position* pos, int n, int point_idx, int k, double d) { // Checks if a point satisfies the proximity criteria
    int neighbors = count_neighbors(pos, n, point_idx, d);
    return (neighbors >= k);
}

int find_trio(Position* pos, int n, int k, double d, int* trio) { // Finds 3 points that satisfy the proximity criteria
    int found = 0;
    int count = 0;
    
    // Search through all points sequentially
    for (int i = 0; i < n && !found; i++) {
        if (check_criteria(pos, n, i, k, d)) {
            trio[count] = i;
            count++;
            if (count == 3) {
                found = 1;  // Found our trio, can stop searching
                break;
            }
        }
    }
    return found;
}


int main() {
    int n, k, tcount;
    double d;
    Point points[MAX_POINTS];
    
    // Read input file
    FILE* file = fopen("input.txt", "r");
    if (!file) {
        printf("Cannot open input.txt\n");
        return 1;
    }
    
    // Read first line: N K D TCount
    fscanf(file, "%d %d %lf %d", &n, &k, &d, &tcount);
    
    // Read point parameters for each of the N points
    for (int i = 0; i < n; i++) {
        fscanf(file, "%d %lf %lf %lf %lf", &points[i].id, &points[i].x1, 
               &points[i].x2, &points[i].a, &points[i].b);
    }
    fclose(file);
    
    clock_t start = clock();
    int found_count = 0;
    
    // Process all time steps
    for (int t_idx = 0; t_idx <= tcount; t_idx++) {
        // Calculate current t value: t = 2*i/tCount - 1, where i = t_idx
        double t = 2.0 * t_idx / tcount - 1.0;
        
        // Compute positions for all points for this t value
        Position pos[MAX_POINTS];
        for (int j = 0; j < n; j++) {
            double x, y;
            compute_position(points[j], t, &x, &y);
            pos[j].x = x;
            pos[j].y = y;
            pos[j].id = points[j].id;
        }
        
        // Find 3 points that satisfy the proximity criteria for this t value
        int trio[3];
        if (find_trio(pos, n, k, d, trio)) {
            found_count++;
            FILE* outfile = fopen("output.txt", "a");
            if (outfile) {
                fprintf(outfile, 
                    "Points %d, %d, %d satisfy criteria at t = %.4f\n",
                    trio[0], trio[1], trio[2], t);
                fclose(outfile);
            }
        }
    }
    
    // Write message if no points found
    if (found_count == 0) {
        FILE* outfile = fopen("output.txt", "a");
        if (outfile) {
            fprintf(outfile, "There were no 3 points found for any t.\n");
            fclose(outfile);
        }
    }
    
    // Measure and print execution time
    clock_t end = clock();
    double elapsed_secs = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Total elapsed time (serial): %.6f seconds\n", elapsed_secs);
    printf("Total trios found: %d\n", found_count);
    
    return 0;
}

