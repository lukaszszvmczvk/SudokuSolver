#include <cmath>
#include <cstdio>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "kernel.cuh"

__global__ void BFS(int* old_boards, int* new_boards, int* board_index,
	int* empty_spaces, int* empty_spaces_count, int total_boards)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

	while (index < total_boards)
	{
		bool empty_place_found = false;
		for (int i = index * N * N; i < index * N * N + N * N; ++i)
		{
			if (old_boards[i] == 0)
			{
				empty_place_found = true;
				int cell_index = i - N * N * index;
				int row = cell_index / N;
				int column = cell_index % N;

				for (int j = 1; j <= N; ++j)
				{
					bool isOkay = true;

					// check row
					for (int c = 0; c < N; ++c)
					{
						if (old_boards[index * N * N + N * row + c] == j)
						{
							isOkay = false;
							break;
						}
					}

					// check column
					for (int r = 0; r < N; ++r)
					{
						if (old_boards[index * N * N + column + r * N] == j)
						{
							isOkay = false;
							break;
						}
					}

					// check box
					for (int r = 3 * (row / 3); r < 3; r++) 
					{
						for (int c = 3 * (column / 3); c < 3; c++)
						{
							if (old_boards[r * N + c + N * N * index] == j) 
							{
								isOkay = 0;
								break;
							}
						}
					}

					if (isOkay)
					{
						int next_board_index = atomicAdd(board_index, 1);
						int empty = 0;
						for (int r = 0; r < N; ++r)
						{
							for (int c = 0; c < N; ++c)
							{
								new_boards[next_board_index * N * N + r * N + c] = old_boards[next_board_index * N * N + r * N + c];
								if (old_boards[next_board_index * N * N + r * N + c] == 0 && (r != row || c != column))
								{
									++empty;
								}

							}
						}
						empty_spaces_count[next_board_index] = empty;
						new_boards[next_board_index * N * N + row * N + column] = j;
					}
				}
			}

			if (empty_place_found)
				break;
		}
		index += gridDim.x * blockDim.x;
	}
}

void kernel_BFS(int* old_boards, int* new_boards, int* board_index,
	int* empty_spaces, int* empty_spaces_count, int total_boards)
{
	BFS <<< max_blocks, threads_count >>> (old_boards, new_boards, board_index, empty_spaces, empty_spaces_count, total_boards);
	cudaDeviceSynchronize();
}

void kernel_DFS()
{

}