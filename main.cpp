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

#define N 9
#define threads_count 256
#define max_blocks 32
#define max_boards 1000000
#define max_boards_size N*N*max_boards
#define iterations 20



void load(std::string filename, int* board);
void print_board(int* board);

int main()
{
    // load and initalize board
	int* board = new int[N * N];
    std::string filename;
    std::cout << "Podaj nazwe pliku z sudoku:\n";
    std::cin >> filename;

    load(filename, board);

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


	return 0;
}

// function to load board from txt file
void load(std::string filename, int* board)
{
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Unable to open file: " << filename << std::endl;
        return;
    }

    for (int i = 0; i < N * N; ++i) {
        if (!(file >> board[i])) {
            std::cerr << "Error reading from file." << std::endl;
            file.close();
            return;
        }
    }

    file.close();
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