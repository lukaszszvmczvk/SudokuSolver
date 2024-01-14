#include "kernel.cuh"
#include <iostream>

__global__ void BFS(unsigned short* old_boards, unsigned short* new_boards, int* board_index,
	int* empty_spaces, int* empty_spaces_count, int boards_count, __int16* old_validators, __int16* new_validators)
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
				int e_id = 0;
				for (int j = 0; j < N * N; ++j)
				{
					new_boards[current_board * N * N + j] = old_boards[board_start + j];
					if (j < validator_size)
					{
						new_validators[current_board * validator_size + j] = old_validators[index * validator_size + j];
					}
					if (new_boards[current_board * N * N + j] == 0 && (j / N != row || j % N != column))
					{
						empty_spaces[current_board * N * N + e_id] = j;
						e_id++;
					}
				}

				empty_spaces_count[current_board] = e_id;

				new_boards[current_board * N * N + empty_index - board_start] = value;
				new_validators[current_board * validator_size + row] |= (1 << value);
				new_validators[current_board * validator_size + N + column] |= (1 << value);
				new_validators[current_board * validator_size + 2 * N + subboard] |= (1 << value);
			}
		}

		index += gridDim.x * blockDim.x;
	}
}

__global__ void DFS(unsigned short* boards, __int16* validators, int boards_count, int* empty_spaces, int* empty_spaces_count, int* sol_found, unsigned short* sol)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;

	unsigned short* current_board;
	int* current_empty_spaces;
	__int16* currentValidators;
	int current_empty_spaces_count;

	while ((*sol_found) == 0 && index < boards_count)
	{
		int empty_index = 0;

		current_board = boards + index * N * N;
		current_empty_spaces = empty_spaces + index * N * N;
		current_empty_spaces_count = empty_spaces_count[index];
		currentValidators = validators + index * validator_size;

		while ((empty_index >= 0) && (empty_index < current_empty_spaces_count))
		{
			int cell_id = current_empty_spaces[empty_index];

			int row = cell_id / N;
			int column = cell_id % N;
			int subboard = (row / 3) * 3 + (column / 3);

			bool flag = false;
			for (int value = current_board[cell_id] + 1; value <= N; ++value)
			{
				int row_flag = (1 << value) & (currentValidators[row]);
				int column_flag = (1 << value) & (currentValidators[N + column]);;
				int subboard_flag = (1 << value) & (currentValidators[2 * N + subboard]);

				if (row_flag == 0 && column_flag == 0 && subboard_flag == 0)
				{
					flag = true;

					current_board[cell_id] = value;
					currentValidators[row] |= (1 << value);
					currentValidators[N + column] |= (1 << value);
					currentValidators[2 * N + subboard] |= (1 << value);
					empty_index++;
					break;

				}
			}

			if (!flag)
			{
				current_board[cell_id] = 0;
				empty_index--;

				if (empty_index >= 0)
				{
					cell_id = current_empty_spaces[empty_index];

					unsigned short value = current_board[cell_id];
					row = cell_id / N;
					column = cell_id % N;
					subboard = (row / 3) * 3 + (column / 3);

					currentValidators[row] &= ~(1 << value);
					currentValidators[N + column] &= ~(1 << value);
					currentValidators[2 * N + subboard] &= ~(1 << value);
				}
			}
		}

		if (empty_index == current_empty_spaces_count)
		{
			*sol_found = 1;

			for (int i = 0; i < N * N; i++) 
			{
				sol[i] = current_board[i];
			}
		}

		index += gridDim.x * blockDim.x;
	}
}

void kernel_BFS(unsigned short* old_boards, unsigned short* new_boards, int* board_index,
	int* empty_spaces, int* empty_spaces_count, int boards_count, __int16* old_validators, __int16* new_validators)
{
	BFS <<< blocks_count, threads_count >>> (old_boards, new_boards, board_index, 
		empty_spaces, empty_spaces_count, boards_count, old_validators, new_validators);
	cudaDeviceSynchronize();
}

void kernel_DFS(unsigned short* boards, __int16* validators, int boards_count, int* empty_spaces, int* empty_spaces_count, int* sol_found, unsigned short* sol)
{
	DFS << < blocks_count, threads_count >> > (boards, validators, boards_count, empty_spaces, empty_spaces_count, sol_found, sol);
	cudaDeviceSynchronize();
}