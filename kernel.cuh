#include "utils.h"
#include <cmath>
#include <cstdio>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <fstream>
#include <cstring>
#include <fstream>
#include <cuda_runtime.h>
#include <algorithm>
#include <curand.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

void kernel_BFS(unsigned short* old_boards, unsigned short* new_boards, int* board_index,
	unsigned short* empty_spaces, int boards_count, __int16* old_validators, __int16* new_validators);

void kernel_DFS(unsigned short* boards, __int16* validators, int boards_count, unsigned short* empty_spaces,
	unsigned short empty_spaces_count, int* sol_found, unsigned short* sol);
