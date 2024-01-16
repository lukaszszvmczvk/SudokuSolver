#include "kernel.cuh"


bool load(std::string filename, unsigned short* board);
void print_board(unsigned short* board);
int run_bfs(unsigned short* prev_boards, unsigned short* new_boards, int* board_index,
    int boards_count, __int16* old_validators, __int16* new_validators, unsigned short* empty_cells, unsigned short* empty_cells_count);
void initialize_validators(unsigned short* board, __int16 validators[]);

int main()
{
    cudaSetDevice(0);
    std::chrono::steady_clock::time_point time_start, time_stop;
    // load and initalize board
    unsigned short* board = new unsigned short[N * N];
    std::string filename;
    std::cout << "Podaj nazwe pliku z sudoku:\n";
    std::cin >> filename;

    if (load(filename, board) == false)
    {
        printf("Taki plik nie istnieje\n");
        return 0;
    }

    std::cout << "Loaded board:\n";
    print_board(board);

    time_start = std::chrono::high_resolution_clock::now();
#pragma region Initialize memory for bfs and dfs

        // initialize variables used in bfs and dfs
        unsigned short* new_boards;
        unsigned short* prev_boards;
        unsigned short* empty_cells;
        unsigned short* empty_cells_count;
        int* board_index;
        __int16* old_validators;
        __int16* new_validators;
        int* solution_found;
        unsigned short* solution_board;

        // allocate memory
        cudaError_t cudaStatus = cudaMalloc(&new_boards, max_boards_size * sizeof(unsigned short));
        if (cudaStatus != cudaSuccess)
            fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaStatus = cudaMalloc(&prev_boards, max_boards_size * sizeof(unsigned short));
        if (cudaStatus != cudaSuccess)
            fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaStatus = cudaMalloc(&empty_cells, N * N * sizeof(unsigned short));
        if (cudaStatus != cudaSuccess)
            fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaStatus = cudaMalloc(&empty_cells_count, sizeof(unsigned short));
        if (cudaStatus != cudaSuccess)
            fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaStatus = cudaMalloc(&board_index, sizeof(int));
        if (cudaStatus != cudaSuccess)
            fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaStatus = cudaMalloc(&old_validators, max_boards_size * sizeof(__int16));
        if (cudaStatus != cudaSuccess)
            fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaStatus = cudaMalloc(&new_validators, max_boards_size * sizeof(__int16));
        if (cudaStatus != cudaSuccess)
            fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaStatus = cudaMalloc(&solution_found, sizeof(int));
        if (cudaStatus != cudaSuccess)
            fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaStatus = cudaMalloc(&solution_board, N * N * sizeof(unsigned short));
        if (cudaStatus != cudaSuccess)
            fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));

        // memset memory
        cudaStatus = cudaMemset(new_boards, 0, max_boards_size * sizeof(unsigned short));
        if (cudaStatus != cudaSuccess)
            fprintf(stderr, "cudaMemset failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaStatus = cudaMemset(prev_boards, 0, max_boards_size * sizeof(unsigned short));
        if (cudaStatus != cudaSuccess)
            fprintf(stderr, "cudaMemset failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaStatus = cudaMemset(board_index, 0, sizeof(int));
        if (cudaStatus != cudaSuccess)
            fprintf(stderr, "cudaMemset failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaStatus = cudaMemset(empty_cells_count, 0, sizeof(unsigned short));
        if (cudaStatus != cudaSuccess)
            fprintf(stderr, "cudaMemset failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaStatus = cudaMemset(old_validators, 0, max_boards_size * sizeof(__int16));
        if (cudaStatus != cudaSuccess)
            fprintf(stderr, "cudaMemset failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaStatus = cudaMemset(new_validators, 0, max_boards_size * sizeof(__int16));
        if (cudaStatus != cudaSuccess)
            fprintf(stderr, "cudaMemset failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaStatus = cudaMemset(solution_found, 0, sizeof(bool));
        if (cudaStatus != cudaSuccess)
            fprintf(stderr, "cudaMemset failed: %s\n", cudaGetErrorString(cudaStatus));

        // copy the initial board to the old boards
        cudaMemcpy(prev_boards, board, N * N * sizeof(unsigned short), cudaMemcpyHostToDevice);
        cudaMemcpy(solution_board, board, N * N * sizeof(unsigned short), cudaMemcpyHostToDevice);
        
        // init validators
        __int16 validator[validator_size] = {0};
        initialize_validators(board, validator);
        cudaMemcpy(old_validators, validator, validator_size * sizeof(__int16), cudaMemcpyHostToDevice);

#pragma endregion
    time_stop = std::chrono::high_resolution_clock::now();

    std::cout << "Allocation of memory took: "<< 0.001 * std::chrono::duration_cast<std::chrono::microseconds>(time_stop - time_start).count() << " ms\n\n";

    // run bfs to create boards
    time_start = std::chrono::high_resolution_clock::now();
    int boards_count = run_bfs(prev_boards, new_boards, board_index, 0, old_validators, new_validators, empty_cells, empty_cells_count);
    time_stop = std::chrono::high_resolution_clock::now();
    printf("Number of boards found in bfs after %d iterations: %d\n", iterations, boards_count);
    auto bfs_time = 0.001 * std::chrono::duration_cast<std::chrono::microseconds>(time_stop - time_start).count();
    std::cout << "BFS took: " << bfs_time << " ms\n\n";

    // run dfs to solve boards
    time_start = std::chrono::high_resolution_clock::now();
    kernel_DFS(new_boards, new_validators, boards_count, empty_cells, empty_cells_count, solution_found, solution_board);
    time_stop = std::chrono::high_resolution_clock::now();
    auto dfs_time = 0.001 * std::chrono::duration_cast<std::chrono::microseconds>(time_stop - time_start).count();
    std::cout << "DFS took: " << dfs_time << " ms\n\n";

    // copy solution to cpu
    unsigned short solution_board_cpu[N * N];
    memset(solution_board_cpu, 0, N * N * sizeof(unsigned short));
    cudaMemcpy(solution_board_cpu, solution_board, N * N * sizeof(unsigned short), cudaMemcpyDeviceToHost);

    // print solution
    std::cout << "Solution board:\n";
    print_board(solution_board_cpu);
    std::cout << "\nAlgorithm took: " << dfs_time + bfs_time << " ms\n\n";


    // free memory
    cudaFree(empty_cells);
    cudaFree(new_boards);
    cudaFree(prev_boards);
    cudaFree(board_index);
    cudaFree(old_validators);
    cudaFree(new_validators);

	return 0;
}

// function to load board from txt file
bool load(std::string filename, unsigned short* board)
{
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Unable to open file: " << filename << std::endl;
        return false;
    }

    for (int i = 0; i < N * N; ++i) {
        if (!(file >> board[i])) {
            std::cerr << "Error reading from file." << std::endl;
            file.close();
            return false;
        }
    }

    file.close();
    return true;
}

// function to print board on console
void print_board(unsigned short* board)
{
    for (int i = 0; i < N; ++i)
    {
        if(i%3 == 0)
            printf("-------------------------------\n");
        for (int j = 0; j < N; ++j)
        {
            if(j%3 == 0)
                printf("|");
            printf(" %d ", board[i * N + j]);
        }
        printf("|\n");
    }
    printf("-------------------------------\n\n");

}

// function to run bfs
int run_bfs(unsigned short* prev_boards, unsigned short* new_boards, int* board_index,
    int boards_count, __int16* old_validators, __int16* new_validators, unsigned short* empty_cells, unsigned short* empty_cells_count)
{
    for (int i = 0; i < iterations; ++i)
    {
        cudaMemcpy(&boards_count, board_index, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemset(board_index, 0, sizeof(int));

        if (boards_count == 0)
            boards_count = 1;

        bool is_last = (i == iterations - 1);

        if (i % 2 == 0)
        {
            kernel_BFS(prev_boards, new_boards, board_index, boards_count, old_validators, new_validators, empty_cells, empty_cells_count, is_last);
        }
        else
        {
            kernel_BFS(new_boards, prev_boards, board_index, boards_count, new_validators, old_validators, empty_cells, empty_cells_count, is_last);
        }
    }


    cudaMemcpy(&boards_count, board_index, sizeof(int), cudaMemcpyDeviceToHost);

    if (iterations % 2 == 0)
    {
        new_boards = prev_boards;
        new_validators = old_validators;
    }
    return boards_count;
}

// init validators
void initialize_validators(unsigned short* board, __int16 validators[])
{
    // validate rows
    for (int i = 0; i < N; ++i)
    {
        int row = 0;
        for (int j = 0; j < N; ++j)
        {
            int value = board[i * N + j];
            if (value != 0)
            {
                row |= (1 << value);
            }
        }
        validators[i] = row;
    }

    // validate columns
    for (int i = 0; i < N; ++i)
    {
        int column = 0;
        for (int j = 0; j < N; ++j)
        {
            int value = board[i + N * j];
            if (value != 0)
            {
                column |= (1 << value);
            }
        }
        validators[N + i] = column;
    }

    // validate subboard
    for (int i = 0; i < N; ++i)
    {
        int subboard = 0;
        int start = ((i / 3) * 3) * N + ((i % 3) * 3);
        for (int j = 0; j < N; ++j)
        {
            int r = j / 3;
            int c = j % 3;
            int value = board[start + r * N + c];
            if (value != 0)
            {
                subboard |= (1 << value);
            }

        }
        validators[2 * N + i] = subboard;
    }
}