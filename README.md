To use c language implementation of DTW, create shared object file:

$ cc -fPIC -shared -o libdtw.so dtw.c

Variables for calculating complexity:
`n` := number of sequences
`m` := length of sequence
`k` := number of classes
