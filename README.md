# HST_LAB5
Генeрация матриц с однозначными цифрами
Загрузка сгенерированных данных в файл (input)
Запись данных в буфер с файла и отправка на девайса
Формирование времени выполнения и размера обработанных данных 
Приницип работы
**cudaMalloc - выделение памяти в gpu
**cudaMemcpyAsync - копируем данные из программы в область gpu
Вычислительная функция (kernel)
__global__ void multiplyMatrix(int* matrix, int* res_matrix, int size){
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int column = blockDim.y * blockIdx.y + threadIdx.y;
    if (row < size && column < size){
        int sum = 0;
        for (int rank = 0; rank < size; rank++)
            sum += matrix[row * size + rank] * matrix[rank * size + column];
        res_matrix[row * size + column] = sum;
    }
}
