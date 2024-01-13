#include "kernel.cuh"

__global__ void BFS(int* old_boards, int* new_boards, int* board_index,
	int* empty_spaces, int* empty_spaces_count, int boards_count, int* old_validators, int* new_validators)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	while (index < boards_count)
	{
		int board_start = index * N * N;
		int empty_index = board_start;
		while (empty_index < board_start + N * N)
		{
			if (old_boards[empty_index] == 0)
				break;
			empty_index++;
		}
		if (empty_index == board_start + N * N)
			return;

		int row = (empty_index - board_start) / N;
		int column = (empty_index - board_start) % N;
		int subboard = (row / 3) * 3 + (column / 3);
		for (int value = 1; value <= N; ++value)
		{
			bool flag = true;

			// check row
			int bit = (1 << value) & (old_validators[index * validator_size + row]);
			if (bit != 0)
			{
				flag = false;
			}

			// check column
			bit = (1 << value) & (old_validators[index * validator_size + N + column]);
			if (bit != 0)
			{
				flag = false;
			}

			// check subboard
			bit = (1 << value) & (old_validators[index * validator_size + 2 * N + subboard]);
			if (bit != 0)
			{
				flag = false;
			}

			if (flag)
			{
				int current_board = atomicAdd(board_index, 1);

				for (int j = 0; j < N * N; ++j)
				{
					new_boards[current_board * N * N + j] = old_boards[board_start + j];
					if (j < validator_size)
					{
						new_validators[current_board * validator_size + j] = old_validators[index * validator_size + j];
					}
				}

				new_boards[current_board * N * N + empty_index - board_start] = value;
				new_validators[current_board * validator_size + row] |= (1 << value);
				new_validators[current_board * validator_size + N + column] |= (1 << value);
				new_validators[current_board * validator_size + 2 * N + subboard] |= (1 << value);
			}
		}

		index += gridDim.x * blockDim.x;
	}
}

void kernel_BFS(int* old_boards, int* new_boards, int* board_index,
	int* empty_spaces, int* empty_spaces_count, int boards_count, int* old_validators, int* new_validators)
{
	BFS <<< blocks_count, threads_count >>> (old_boards, new_boards, board_index, 
		empty_spaces, empty_spaces_count, boards_count, old_validators, new_validators);
	cudaDeviceSynchronize();
}

void kernel_DFS()
{

}