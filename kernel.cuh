#include "utils.h"
#include <cmath>
#include <cstdio>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

void kernel_BFS(int* old_boards, int* new_boards, int* board_index,
	int* empty_spaces, int* empty_spaces_count, int boards_count, int* old_validators, int* new_validators);
void kernel_DFS();
