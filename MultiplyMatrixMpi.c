#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <time.h>
#include "mpi.h"
#define MEGA 1024 * 1024

void multiplyMatrix(int* matrix, int* res_matrix, int size);

int main(int argc, char *argv[]){
    float start_time = (float)clock();
    float end_time;
    int rc, rank, number_of_tasks;;
    if ((rc = MPI_Init(&argc, &argv)) != MPI_SUCCESS){
        printf("Error starting MPI programm, Terminating!\n");
        MPI_Abort(MPI_COMM_WORLD, rc);
    }
    MPI_Comm_size(MPI_COMM_WORLD, &number_of_tasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0){
        int** matrices;
        int* rangs_of_matrices;
        int size_of_data = 0;
        int count_of_matrices;
        FILE* fp;
        fp = fopen("input", "rb");
        fread(&count_of_matrices, sizeof(int), 1, fp);
        matrices = (int**) malloc(count_of_matrices * sizeof(int*));
        rangs_of_matrices = (int*) malloc(count_of_matrices * sizeof(int));
        for (int index_of_matrix = 0; index_of_matrix < count_of_matrices; index_of_matrix++){
            int rang;
            fread(&rang, sizeof(int), 1, fp);
            rangs_of_matrices[index_of_matrix] = rang;
            matrices[index_of_matrix] = (int*) malloc(rang * rang * sizeof(int));
            for (int row = 0; row < rang; row++)
                for (int column = 0; column < rang; column++)
                    fread(&matrices[index_of_matrix][row * rang + column], sizeof(int), 1, fp);
            size_of_data += rang * rang * sizeof(int);
        }
        fclose(fp);
        MPI_Status status;
        for (int number_of_rank = 1; number_of_rank < count_of_matrices % (number_of_tasks - 1) + 1; number_of_rank++)
        {
            int message = count_of_matrices / (number_of_tasks - 1) + 1;
            MPI_Send(&message, 1, MPI_INT, number_of_rank, 0, MPI_COMM_WORLD);
        }
        for (int number_of_rank = count_of_matrices % (number_of_tasks - 1) + 1; number_of_rank < number_of_tasks; number_of_rank++)
        {
            int message = count_of_matrices / (number_of_tasks - 1);
            MPI_Send(&message, 1, MPI_INT, number_of_rank, 0, MPI_COMM_WORLD);
        }
        for (int index_of_set = 0; index_of_set < count_of_matrices / (number_of_tasks - 1) * (number_of_tasks - 1); index_of_set += (number_of_tasks - 1))
        {
            for (int number_of_rank = 1; number_of_rank < number_of_tasks; number_of_rank++)
            {
                MPI_Send(&(rangs_of_matrices[index_of_set + number_of_rank - 1]), 1, MPI_INT, number_of_rank, 0, MPI_COMM_WORLD);
                MPI_Send(matrices[index_of_set + number_of_rank - 1], rangs_of_matrices[index_of_set + number_of_rank - 1] * rangs_of_matrices[index_of_set + number_of_rank - 1], MPI_INT, number_of_rank, 0, MPI_COMM_WORLD);
            }
            for (int number_of_rank = 1; number_of_rank < number_of_tasks; number_of_rank++)
            {
                MPI_Status status;
                MPI_Recv(matrices[index_of_set + number_of_rank - 1], rangs_of_matrices[index_of_set + number_of_rank - 1] * rangs_of_matrices[index_of_set + number_of_rank - 1], MPI_INT, number_of_rank, 0, MPI_COMM_WORLD, &status);
            }
        }
        for (int number_of_rank = 1; number_of_rank < count_of_matrices % (number_of_tasks - 1) + 1; number_of_rank++)
        {
            MPI_Send(&(rangs_of_matrices[count_of_matrices / (number_of_tasks - 1) * (number_of_tasks - 1) + number_of_rank - 1]), 1, MPI_INT, number_of_rank, 0, MPI_COMM_WORLD);
            MPI_Send(matrices[count_of_matrices / (number_of_tasks - 1) * (number_of_tasks - 1) + number_of_rank - 1], rangs_of_matrices[count_of_matrices / (number_of_tasks - 1) * (number_of_tasks - 1) + number_of_rank - 1] * rangs_of_matrices[count_of_matrices / (number_of_tasks - 1) * (number_of_tasks - 1) + number_of_rank - 1], MPI_INT, number_of_rank, 0, MPI_COMM_WORLD);
        }
        for (int number_of_rank = 1; number_of_rank < count_of_matrices % (number_of_tasks - 1) + 1; number_of_rank++)
        {
            MPI_Status status;
            MPI_Recv(matrices[count_of_matrices / (number_of_tasks - 1) * (number_of_tasks - 1) + number_of_rank - 1], rangs_of_matrices[count_of_matrices / (number_of_tasks - 1) * (number_of_tasks - 1) + number_of_rank - 1] * rangs_of_matrices[count_of_matrices / (number_of_tasks - 1) * (number_of_tasks - 1) + number_of_rank - 1], MPI_INT, number_of_rank, 0, MPI_COMM_WORLD, &status);
        }
        fp = fopen("output.txt", "w");
        fprintf(fp, "%d\n", count_of_matrices);
        for (int index_of_matrix = 0; index_of_matrix < count_of_matrices; index_of_matrix++){
            fprintf(fp, "%d\n", rangs_of_matrices[index_of_matrix]);
            for (int row = 0; row < rangs_of_matrices[index_of_matrix]; row++){
                for (int column = 0; column < rangs_of_matrices[index_of_matrix]; column++)
                    fprintf(fp, "%d ", matrices[index_of_matrix][row *rangs_of_matrices[index_of_matrix] + column]);
                fputs("\n", fp);
            }
        }
        for (int index_of_matrix = 0; index_of_matrix < count_of_matrices; index_of_matrix++)
            free(matrices[index_of_matrix]);
        free(matrices);
        free(rangs_of_matrices);
        end_time = ((float)clock()) - start_time;
        fprintf(fp, "Count time: %f s\n", end_time / 1000000);
        fprintf(fp, "Size of data: %f Mb\n", (float)size_of_data / (MEGA));
        fclose(fp);
        printf("Success!\n");
    }
    else{
        MPI_Status status;
        int count_of_matrices = 0;
        MPI_Recv(&count_of_matrices, sizeof(int), MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        for (int index_of_matrix = 0; index_of_matrix < count_of_matrices; index_of_matrix++){
            int rank_of_matrix = 0;
            MPI_Recv(&rank_of_matrix, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
            int* matrix = (int*) malloc(rank_of_matrix * rank_of_matrix * sizeof(int));
            int* buffer_matrix = (int*) malloc(rank_of_matrix * rank_of_matrix * sizeof(int));
            MPI_Recv(buffer_matrix, rank_of_matrix * rank_of_matrix, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
            multiplyMatrix(buffer_matrix, matrix, rank_of_matrix);
            MPI_Send(matrix, rank_of_matrix * rank_of_matrix, MPI_INT, 0, 0, MPI_COMM_WORLD);
            free(buffer_matrix);
            free(matrix);
        }
    }
    MPI_Finalize();
    return 0;
}
