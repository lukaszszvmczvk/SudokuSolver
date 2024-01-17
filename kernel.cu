#include "kernel.cuh"

__global__ void BFS(unsigned short* old_boards, unsigned short* new_boards, int* board_index, int boards_count, __int16* old_validators, 
	__int16* new_validators, unsigned short* empty_cells, unsigned short* empty_cells_count, bool is_last)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

	while (index < boards_count && boards_count < max_boards)
	{
		// get board_start index
		int board_start = index * N * N;
		int empty_index = board_start;

		// find first empty index in board
		while (empty_index < board_start + N * N)
		{
			if (old_boards[empty_index] == 0)
			{
				// get column, row and subboard of current cell
				int row = (empty_index - board_start) / N;
				int column = (empty_index - board_start) % N;
				int subboard = (row / 3) * 3 + (column / 3);

				// create boards with correct values
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

					// if correct then add board to new_boards
					if (flag)
					{
						// get current_board and update shared board_index
						int current_board = atomicAdd(board_index, 1);
						int e_id = 0;
						for (int j = 0; j < N * N; ++j)
						{
							// update new_boards
							new_boards[current_board * N * N + j] = old_boards[board_start + j];
							if (j < validator_size)
							{
								// update validators
								new_validators[current_board * validator_size + j] = old_validators[index * validator_size + j];
							}
							if (is_last && new_boards[current_board * N * N + j] == 0 && (j / N != row || j % N != column))
							{
								// update empty spaces used in DFS
								empty_cells[e_id] = j;
								e_id++;
							}
						}

						// assign empty cells count
						*empty_cells_count = e_id;

						// assign value to empty cell
						new_boards[current_board * N * N + empty_index - board_start] = value;

						// update validators with added value
						new_validators[current_board * validator_size + row] |= (1 << value);
						new_validators[current_board * validator_size + N + column] |= (1 << value);
						new_validators[current_board * validator_size + 2 * N + subboard] |= (1 << value);
					}
				}

				// empty cell found
				break;
			}
			else
			{
				empty_index++;
			}
		}

		index += gridDim.x * blockDim.x;
	}
}

__global__ void DFS(unsigned short* boards, __int16* validators, int boards_count, unsigned short* empty_cells, unsigned short* empty_cells_count, int* sol_found, unsigned short* sol)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;

	while ((*sol_found) == 0 && index < boards_count)
	{
		// initialize current index in empty cells array
		int empty_index = 0;

		// get current board and validators
		unsigned short* current_board = boards + index * N * N;
		__int16* currentValidators = validators + index * validator_size;

		// backtrack solution 
		// end while if there is no solution or solution is found
		while (empty_index >= 0 && empty_index < *empty_cells_count)
		{
			// current empty cell index 
			int cell_id = empty_cells[empty_index];

			// compute row, column and subboard of current cell
			int row = cell_id / N;
			int column = cell_id % N;
			int subboard = (row / 3) * 3 + (column / 3);

			// initialize flag to check if there is a possible value to put
			bool flag = false;

			for (int value = current_board[cell_id] + 1; value <= N; ++value)
			{
				// check if value matches conditions
				int row_flag = (1 << value) & (currentValidators[row]);
				int column_flag = (1 << value) & (currentValidators[N + column]);;
				int subboard_flag = (1 << value) & (currentValidators[2 * N + subboard]);

				flag = row_flag == 0 && column_flag == 0 && subboard_flag == 0;

				// if value is correct update board and validators
				if (flag)
				{
					current_board[cell_id] = value;
					currentValidators[row] |= (1 << value);
					currentValidators[N + column] |= (1 << value);
					currentValidators[2 * N + subboard] |= (1 << value);

					empty_index++;
					break;
				}
			}

			// if there is not possible value to put
			if (!flag)
			{
				current_board[cell_id] = 0;
				empty_index--;

				// if there is still posibility to find solution then update board and validators
				if (empty_index >= 0)
				{
					cell_id = empty_cells[empty_index];

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

		// copy board if solution found
		if (empty_index == *empty_cells_count)
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

void kernel_BFS(unsigned short* old_boards, unsigned short* new_boards, int* board_index, int boards_count, __int16* old_validators, __int16* new_validators, unsigned short* empty_cells, unsigned short* empty_cells_count, bool is_last)
{
	BFS <<< blocks_count, threads_count >>> (old_boards, new_boards, board_index, 
		boards_count, old_validators, new_validators, empty_cells, empty_cells_count, is_last);
	cudaDeviceSynchronize();
}

void kernel_DFS(unsigned short* boards, __int16* validators, int boards_count, unsigned short* empty_cells, unsigned short* empty_cells_count, int* sol_found, unsigned short* sol)
{
	DFS << < blocks_count, threads_count >> > (boards, validators, boards_count, empty_cells, empty_cells_count, sol_found, sol);
	cudaDeviceSynchronize();
}