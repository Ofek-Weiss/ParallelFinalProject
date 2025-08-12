# Parallel Implementation of Proximity Criteria
## Final Project - Course 10324, Parallel and Distributed Computation
### Spring Semester 2025

**Student:** Ofek Weiss  
**Implementation:** MPI + OpenMP Hybrid Parallelization

## Problem Description

Given a set of N points in a two-dimensional plane, where each point P has coordinates (x, y) defined by:
- x = ((x2 + x1) / 2) * cos(t*π/2) + (x2 - x1) / 2
- y = a*x + b

Where (x1, x2, a, b) are predefined parameters for each point.

**Proximity Criteria:** A point P satisfies the criteria if there exist at least K points in the set with distance from P less than a given value D.

**Goal:** For tCount + 1 values of t (t = 2*i/tCount - 1, i = 0,1,2,...,tCount), find if there exist at least 3 points that satisfy the Proximity Criteria.

## Parallelization Strategy

### Architecture Choice: MPI + OpenMP Hybrid

**Rationale:**
1. **MPI (Message Passing Interface):** 
   - Distributes work across multiple nodes/computers
   - Each MPI process handles a subset of t values
   - Enables scaling across multiple machines in VLAB environment

2. **OpenMP (Open Multi-Processing):**
   - Provides shared-memory parallelism within each MPI process
   - Parallelizes the computation of point positions and neighbor counting
   - Efficient for fine-grained parallelism

### Load Balancing Strategy

**MPI Level:**
- Work distribution: Each process gets approximately (tCount + 1) / num_processes t values
- Extra work distributed to first processes: `start_t = rank * work_per_process + (rank < extra_work ? rank : extra_work)`

**OpenMP Level:**
- Dynamic scheduling for neighbor counting: `#pragma omp parallel for schedule(dynamic)`
- Thread count: `omp_set_num_threads(omp_get_num_procs() / size)` - divides available cores among MPI processes

### Complexity Analysis

**Serial Algorithm:**
- Time Complexity: O(tCount * N²) - for each t value, check all N points against all other N points

**Parallel Algorithm:**
- Time Complexity: O((tCount/P) * N²/T) where P = MPI processes, T = OpenMP threads
- Expected Speedup: P * T (theoretical maximum)

## Implementation Details

### Key Data Structures

```c
typedef struct {
    int id;
    double x1, x2, a, b;  // Point parameters
} Point;

typedef struct {
    int id;
    double x, y;          // Computed position
} Position;
```

### Core Functions

1. **`compute_position()`**: Computes (x,y) coordinates for a point given parameter t
2. **`count_neighbors()`**: Counts neighbors within distance D (stops when count reaches K)
3. **`find_trio()`**: Finds 3 points satisfying proximity criteria using OpenMP parallelization

### Parallelization Points

1. **MPI Distribution**: Each process handles different t values
2. **OpenMP Position Computation**: `#pragma omp parallel for` for computing all point positions
3. **OpenMP Neighbor Counting**: `#pragma omp parallel for schedule(dynamic)` for finding qualifying points
4. **Critical Section**: `#pragma omp critical` for safely collecting results

### Communication Pattern

1. **Broadcast**: Rank 0 broadcasts input parameters to all processes
2. **Reduce**: Sum up found counts from all processes
3. **Gather**: Collect result strings from all processes to rank 0

## Performance Optimizations

1. **Early Termination**: Stop counting neighbors when K neighbors are found
2. **Dynamic Scheduling**: OpenMP dynamic scheduling for better load balancing
4. **Communication Optimization**: Batch result collection using MPI_Gather

## Input/Output Format

**Input (input.txt):**
```
N K D TCount
id x1 x2 a b
id x1 x2 a b
...
```

**Output (output.txt):**
```
Points id1, id2, id3 satisfy criteria at t = t_value
...
There were no 3 points found for any t.
```

## Build and Execution

```bash
# Build both serial and parallel versions
make

# Run serial version
./serial

# Run parallel version (example with # processes)
mpiexec -np '#' ./parallel

# Run parallel version with two different computers from two different pools
# The two ip's of the two computers must be written in the "myhost.txt" file
# You can get the computer ip by writting on terminal "Hostname -I"
mpiexec -np '#' -hostfile myhost ./parallel
```

## Testing and Validation

- Verified correctness by comparing serial and parallel outputs
- Tested on VLAB environment with multiple nodes (two different computers from 2 different pools)
- Measured performance improvement over serial implementation
- Validated edge cases (no solutions, single solution, multiple solutions)