#include "kernel.cuh"


bool load(std::string filename, unsigned short* board);
void print_board(unsigned short* board);
int run_bfs(unsigned short* prev_boards, unsigned short* new_boards, int* board_index,
    int* empty_spaces, int* empty_spaces_count, int boards_count, __int16* old_validators, __int16* new_validators);
void initialize_validators(unsigned short* board, __int16 validators[]);

int main()
{
    cudaSetDevice(0);
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


#pragma region Initialize memory

        // the boards after the next iteration of breadth first search
        unsigned short* new_boards;
        // the previous boards, which formthe frontier of the breadth first search
        unsigned short* prev_boards;
        // stores the location of the empty spaces in the boards
        int* empty_spaces;
        // stores the number of empty spaces in each board
        int* empty_space_count;
        // where to store the next new board generated
        int* board_index;
        __int16* old_validators;
        __int16* new_validators;

        // allocate memory
        cudaError_t cudaStatus = cudaMalloc(&new_boards, max_boards_size * sizeof(unsigned short));
        if (cudaStatus != cudaSuccess)
            fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaStatus = cudaMalloc(&prev_boards, max_boards_size * sizeof(unsigned short));
        if (cudaStatus != cudaSuccess)
            fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaStatus = cudaMalloc(&empty_spaces, max_boards_size * sizeof(int));
        if (cudaStatus != cudaSuccess)
            fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaStatus = cudaMalloc(&empty_space_count, max_boards_size * sizeof(int));
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


        cudaStatus = cudaMemset(new_boards, 0, max_boards_size * sizeof(unsigned short));
        if (cudaStatus != cudaSuccess)
            fprintf(stderr, "cudaMemset failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaStatus = cudaMemset(prev_boards, 0, max_boards_size * sizeof(unsigned short));
        if (cudaStatus != cudaSuccess)
            fprintf(stderr, "cudaMemset failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaStatus = cudaMemset(board_index, 0, sizeof(int));
        if (cudaStatus != cudaSuccess)
            fprintf(stderr, "cudaMemset failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaStatus = cudaMemset(old_validators, 0, max_boards_size * sizeof(__int16));
        if (cudaStatus != cudaSuccess)
            fprintf(stderr, "cudaMemset failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaStatus = cudaMemset(new_validators, 0, max_boards_size * sizeof(__int16));
        if (cudaStatus != cudaSuccess)
            fprintf(stderr, "cudaMemset failed: %s\n", cudaGetErrorString(cudaStatus));

        // copy the initial board to the old boards
        cudaMemcpy(prev_boards, board, N * N * sizeof(unsigned short), cudaMemcpyHostToDevice);
        
        __int16 validator[validator_size] = {0};
        initialize_validators(board, validator);

        cudaMemcpy(old_validators, validator, validator_size * sizeof(__int16), cudaMemcpyHostToDevice);

#pragma endregion

    int boards_count = run_bfs(prev_boards, new_boards, board_index, empty_spaces, empty_space_count, 0, old_validators, new_validators);

    // flag to determine when a solution has been found
    int* dev_finished;
    // output to store solved board in
    int* dev_solved;

    printf("Number of boards found in bfs: %d\n", boards_count);

    print_board(board);

    cudaFree(empty_spaces);
    cudaFree(empty_space_count);
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
        printf("-------------------------------------\n");
        for (int j = 0; j < N; ++j)
        {
            printf("|");
            printf(" %d ", board[i * N + j]);
        }
        printf("|\n");
    }
    printf("-------------------------------------\n\n");
}

// function to run bfs
int run_bfs(unsigned short* prev_boards, unsigned short* new_boards, int* board_index,
    int* empty_spaces, int* empty_spaces_count, int boards_count, __int16* old_validators, __int16* new_validators)
{
    for (int i = 0; i < iterations; ++i)
    {
        cudaMemcpy(&boards_count, board_index, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemset(board_index, 0, sizeof(int));

        if (boards_count == 0)
            boards_count = 1;

        if (i % 2 == 0)
        {
            kernel_BFS(prev_boards, new_boards, board_index, empty_spaces, empty_spaces_count, boards_count, old_validators, new_validators);
        }
        else
        {
            kernel_BFS(new_boards, prev_boards, board_index, empty_spaces, empty_spaces_count, boards_count, new_validators, old_validators);
        }
    }
    cudaMemcpy(&boards_count, board_index, sizeof(int), cudaMemcpyDeviceToHost);
    return boards_count;
}

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