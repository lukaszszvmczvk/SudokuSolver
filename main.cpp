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
#include <iostream>

#include "kernel.cuh"


bool load(std::string filename, int* board);
void print_board(int* board);
void run_bfs(int* old_boards, int* new_boards, int* board_index,
    int* empty_spaces, int* empty_spaces_count, int total_boards);
void run_dfs(int* boards);
int main()
{
    if (0)
    {
        printf("test");
    }
    // load and initalize board
	int* board = new int[N * N];
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
        int* new_boards;
        // the previous boards, which formthe frontier of the breadth first search
        int* old_boards;
        // stores the location of the empty spaces in the boards
        int* empty_spaces;
        // stores the number of empty spaces in each board
        int* empty_space_count;
        // where to store the next new board generated
        int* board_index;

        // allocate memory
        cudaError_t cudaStatus = cudaMalloc(&new_boards, max_boards_size * sizeof(int));
        if (cudaStatus != cudaSuccess)
            fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaStatus = cudaMalloc(&old_boards, max_boards_size * sizeof(int));
        if (cudaStatus != cudaSuccess)
            fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaStatus = cudaMalloc(&empty_spaces, max_boards_size * sizeof(int));
        if (cudaStatus != cudaSuccess)
            fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaStatus = cudaMalloc(&empty_space_count, max_boards * sizeof(int));
        if (cudaStatus != cudaSuccess)
            fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaStatus = cudaMalloc(&board_index, sizeof(int));
        if (cudaStatus != cudaSuccess)
            fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));


        cudaStatus = cudaMemset(new_boards, 0, max_boards_size * sizeof(int));
        if (cudaStatus != cudaSuccess)
            fprintf(stderr, "cudaMemset failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaStatus = cudaMemset(old_boards, 0, max_boards_size * sizeof(int));
        if (cudaStatus != cudaSuccess)
            fprintf(stderr, "cudaMemset failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaStatus = cudaMemset(board_index, 0, sizeof(int));
        if (cudaStatus != cudaSuccess)
            fprintf(stderr, "cudaMemset failed: %s\n", cudaGetErrorString(cudaStatus));

        // copy the initial board to the old boards
        cudaMemcpy(old_boards, board, N * N * sizeof(int), cudaMemcpyHostToDevice);

#pragma endregion
    
    run_bfs(old_boards, new_boards, board_index, empty_spaces, empty_space_count, 0);

    // flag to determine when a solution has been found
    int* dev_finished;
    // output to store solved board in
    int* dev_solved;

    // allocate memory on the device
    cudaMalloc(&dev_finished, sizeof(int));
    cudaMalloc(&dev_solved, N * N * sizeof(int));

    // initialize memory
    cudaMemset(dev_finished, 0, sizeof(int));
    cudaMemcpy(dev_solved, board, N * N * sizeof(int), cudaMemcpyHostToDevice);


    print_board(board);


    cudaFree(empty_spaces);
    cudaFree(empty_space_count);
    cudaFree(new_boards);
    cudaFree(old_boards);
    cudaFree(board_index);

    cudaFree(dev_finished);
    cudaFree(dev_solved);
	return 0;
}

// function to load board from txt file
bool load(std::string filename, int* board)
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
void print_board(int* board)
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

void initialize_memory(int* new_boards, int* old_boards, int* empty_spaces, int* empty_space_count)
{
    // allocate memory
    cudaError_t cudaStatus = cudaMalloc(&new_boards, max_boards_size * sizeof(int));
    if (cudaStatus != cudaSuccess)
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
    cudaStatus = cudaMalloc(&old_boards, max_boards_size * sizeof(int));
    if (cudaStatus != cudaSuccess)
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
    cudaStatus = cudaMalloc(&empty_spaces, max_boards_size * sizeof(int));
    if (cudaStatus != cudaSuccess)
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
    cudaStatus = cudaMalloc(&empty_space_count, max_boards * sizeof(int));
    if (cudaStatus != cudaSuccess)
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));


    cudaStatus = cudaMemset(new_boards, 0, max_boards_size * sizeof(int));
    if (cudaStatus != cudaSuccess)
        fprintf(stderr, "cudaMemset failed: %s\n", cudaGetErrorString(cudaStatus));
    cudaStatus = cudaMemset(old_boards, 0, max_boards_size * sizeof(int));
    if (cudaStatus != cudaSuccess)
        fprintf(stderr, "cudaMemset failed: %s\n", cudaGetErrorString(cudaStatus));
}

// function to run bfs
void run_bfs(int* old_boards, int* new_boards, int* board_index,
    int* empty_spaces, int* empty_spaces_count, int total_boards)
{
    for (int i = 0; i < iterations; ++i)
    {
        cudaMemcpy(&total_boards, board_index, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemset(board_index, 0, sizeof(int));

        if (total_boards == 0)
            total_boards = 1;

        if (i % 2 == 0)
        {
            kernel_BFS(old_boards, new_boards, board_index, empty_spaces, empty_spaces_count, total_boards);
        }
        else
        {
            kernel_BFS(new_boards, old_boards, board_index, empty_spaces, empty_spaces_count, total_boards);
        }
    }
}