# HST_LAB5
Генeрация матриц с однозначными цифрами
Загрузка сгенерированных данных в файл (input)
Запись данных в буфер с файла и отправка на девайса
Формирование времени выполнения и размера обработанных данных 
Приницип работы
**cudaMalloc - выделение памяти в gpu
**cudaMemcpyAsync - копируем данные из программы в область gpu
Вычислительная функция (расширяющая MPI, то есть вызывается в MPI)
extern "C"  void multiplyMatrix(int* matrix, int* res_matrix, int size){
    int* cuda_matrix;
    gpuErrchk( cudaMalloc((void**)&cuda_matrix, size * size * sizeof(int)) );
    int* cuda_res_matrix;
    gpuErrchk( cudaMalloc((void**)&cuda_res_matrix, size * size * sizeof(int)) );
    gpuErrchk( cudaMemcpy(cuda_matrix, matrix, size * size * sizeof(int), cudaMemcpyHostToDevice) );
    dim3 threadsPerBlock(size, size);
    dim3 numBlocks(size / threadsPerBlock.x, size / threadsPerBlock.y);
    kernel<<<numBlocks, threadsPerBlock>>>(cuda_matrix, cuda_res_matrix, size);
    gpuErrchk( cudaMemcpy(res_matrix, cuda_res_matrix, size * size * sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrchk( cudaFree(cuda_matrix) );
    gpuErrchk( cudaFree(cuda_res_matrix) );
}
Kernel
