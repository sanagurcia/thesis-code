#include <math.h>  // fabs() - absolute value doubles
#include <stdio.h>
#include <stdlib.h>  // malloc()

// Return min of three values
float get_min(float a, float b, float c) {
    float min;
    if ((a < b) && (a < c)) {
        min = a;
    } else if (b < c) {
        min = b;
    } else {
        min = c;
    }
    return min;
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
    D[0][0] = (float)fabs(a[0] - b[0]);  // fabs returns double

    // compute dtw cost for first column
    for (int i = 1; i < a_len; i++) {
        // dtw_cost := current_cost + predecessor_cost
        D[i][0] = (float)fabs(a[i] - b[0]) + D[i - 1][0];
    }

    // compute dtw cost for first row
    for (int j = 1; j < b_len; j++) {
        D[0][j] = (float)fabs(a[0] - b[j]) + D[0][j - 1];
    }

    // cost for all other entries
    for (int k = 1; k < a_len; k++) {
        for (int l = 1; l < b_len; l++) {
            float d0 = D[k - 1][l];      // top
            float d1 = D[k][l - 1];      // left
            float d2 = D[k - 1][l - 1];  // diagonal

            // find min predecessor
            float min = get_min(d0, d1, d2);
            D[k][l] = (float)fabs(a[k] - b[l]) + min;
        }
    }

    float cost = D[a_len - 1][b_len - 1];  // cost in last entry
    free(D);                               // free allocated space
    return cost;
}

// Return warping path from DTW computation
int dtw_path(int a_len, int b_len, float *a, float *b, unsigned short *wp) {
    // D holds accumulated DTW cost
    // P holds min predecessor: 2D array with (a_i, b_j) tuples, implemented as 3D array

    // Allocate D: a_len x b_len (2D) array of floats
    float(*D)[b_len] = malloc(sizeof(float[a_len][b_len]));

    // Allocate P: a_len x b_len x 2 array of ints
    unsigned short(*P)[b_len][2] = malloc(sizeof(unsigned short[a_len][b_len][2]));

    if (D == NULL || P == NULL) {
        return -1;  // malloc failed; abort
    }

    // COMPUTE DTW ARRAYS: D & P
    // init first entry
    D[0][0] = (float)fabs(a[0] - b[0]);  // fabs returns double
    P[0][0][0] = 0;
    P[0][0][1] = 0;

    // compute dtw cost for first column
    for (int i = 1; i < a_len; i++) {
        // dtw_cost := current_cost + predecessor_cost
        D[i][0] = (float)fabs(a[i] - b[0]) + D[i - 1][0];
        P[i][0][0] = i - 1;
        P[i][0][1] = 0;
    }

    // compute dtw cost for first row
    for (int j = 1; j < b_len; j++) {
        D[0][j] = (float)fabs(a[0] - b[j]) + D[0][j - 1];
        P[0][j][0] = 0;
        P[0][j][1] = j - 1;
    }

    // cost for all other entries
    for (int k = 1; k < a_len; k++) {
        for (int l = 1; l < b_len; l++) {
            float d0 = D[k - 1][l];      // top
            float d1 = D[k][l - 1];      // left
            float d2 = D[k - 1][l - 1];  // diagonal

            // find min predecessor value & indices
            float min;
            unsigned short p_a;
            unsigned short p_b;
            if ((d0 < d1) && (d0 < d2)) {
                min = d0;
                p_a = k - 1;
                p_b = l;
            } else if (d1 < d2) {
                min = d1;
                p_a = k;
                p_b = l - 1;
            } else {
                min = d2;
                p_a = k - 1;
                p_b = l - 1;
            }

            D[k][l] = (float)fabs(a[k] - b[l]) + min;
            P[k][l][0] = p_a;
            P[k][l][1] = p_b;
        }
    }

    // EXTRACT WARPING PATH FROM P
    // Indices a_i of seq_a, b_j of seq_b
    unsigned short a_i = a_len - 1;
    unsigned short b_j = b_len - 1;

    // Init first entry in path with last two indices
    wp[0] = a_i;
    wp[1] = b_j;

    int z = 2;
    while (!(a_i == 0 && b_j == 0)) {
        wp[z] = P[a_i][b_j][0];
        wp[z + 1] = P[a_i][b_j][1];
        a_i = wp[z];
        b_j = wp[z + 1];
        z += 2;
    }

    free(D);  // free allocated space
    free(P);
    return z;  // return warping path length
}