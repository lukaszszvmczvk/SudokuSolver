#pragma once

#define N 9
#define threads_count 512
#define blocks_count 256
#define max_boards_size pow(2, 28)
#define max_boards (max_boards_size/81)-1
#define validator_size 27
