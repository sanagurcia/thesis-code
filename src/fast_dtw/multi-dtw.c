#include <math.h>  // sqrtf() - square root float
#include <stdio.h>
#include <stdlib.h>  // malloc()

// a & b are arrays, i & j corresponding indices to these arrays
float local_squared_distance(float *a, float *b, int i, int j) {
    float diff = (float)(a[i] - b[j]);
    return diff * diff;
}

// Return dtw distance between two sequences
float dtw_cost(int a_len, int b_len, float *a, float *b) {
    // D holds accumulated DTW cost
    // Allocate D: a_len x b_len (2D) array of floats
    float(*D)[b_len] = malloc(sizeof(float[a_len][b_len]));
    if (D == NULL) {
        return 0.0;  // malloc failed; abort
    }

    // init first entry
    D[0][0] = local_squared_distance(a, b, 0, 0);

    // compute dtw cost for first column
    for (int i = 1; i < a_len; i++) {
        // dtw_cost := current_cost + predecessor_cost
        D[i][0] = local_squared_distance(a, b, i, 0) + D[i - 1][0];
    }

    // compute dtw cost for first row
    for (int j = 1; j < b_len; j++) {
        D[0][j] = local_squared_distance(a, b, 0, j) + D[0][j - 1];
    }

    // cost for all other entries
    for (int k = 1; k < a_len; k++) {
        for (int l = 1; l < b_len; l++) {
            float d0 = D[k - 1][l];      // top
            float d1 = D[k][l - 1];      // left
            float d2 = D[k - 1][l - 1];  // diagonal

            // find min predecessor
            float min;
            if ((d0 < d1) && (d0 < d2)) {
                min = d0;
            } else if (d1 < d2) {
                min = d1;
            } else {
                min = d2;
            }

            D[k][l] = local_squared_distance(a, b, k, l) + min;
        }
    }

    float cost = D[a_len - 1][b_len - 1];  // cost in last entry
    free(D);                               // free allocated space
    return sqrtf(cost);
}
