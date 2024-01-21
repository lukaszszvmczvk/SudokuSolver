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
#include <chrono>
#include <string>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

void kernel_BFS(unsigned short* old_boards, unsigned short* new_boards, int* board_index, int boards_count, 
	__int16* old_validators, __int16* new_validators, unsigned short* empty_cells, unsigned short* empty_cells_count, int* end_flag);

void kernel_DFS(unsigned short* boards, __int16* validators, int boards_count, unsigned short* empty_cells,
	unsigned short* empty_cells_count, int* sol_found, unsigned short* sol);
