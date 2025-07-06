% 矩阵规模
matrix_sizes = [32, 64, 128, 256, 512, 1024, 2048];

% 执行时间（单位：秒）
serial_time = [0.012, 0.050, 0.200, 1.200, 8.000, 64.000, 512.000];
simd_time = [0.005, 0.020, 0.080, 0.500, 3.200, 28.000, 200.000];
omp_time = [0.004, 0.018, 0.070, 0.450, 2.800, 22.500, 150.000];
cuda_time = [0.0015, 0.0050, 0.018, 0.100, 0.350, 2.000, 15.000];
cuda_omp_simd_time = [0.0013, 0.0043, 0.0155, 0.085, 0.300, 1.500, 10.000];

% 加速比（相对于串行）
simd_speedup = serial_time ./ simd_time;
omp_speedup = serial_time ./ omp_time;
cuda_speedup = serial_time ./ cuda_time;
cuda_omp_simd_speedup = serial_time ./ cuda_omp_simd_time;

% 创建图形
figure;

% 绘制执行时间图
subplot(2, 1, 1);
plot(matrix_sizes, serial_time, 'o-', 'LineWidth', 2, 'MarkerSize', 6);
hold on;
plot(matrix_sizes, simd_time, 's-', 'LineWidth', 2, 'MarkerSize', 6);
plot(matrix_sizes, omp_time, '^-', 'LineWidth', 2, 'MarkerSize', 6);
plot(matrix_sizes, cuda_time, 'd-', 'LineWidth', 2, 'MarkerSize', 6);
plot(matrix_sizes, cuda_omp_simd_time, 'p-', 'LineWidth', 2, 'MarkerSize', 6);

xlabel('矩阵规模 (N)');
ylabel('执行时间 (秒)');
legend({'串行', 'SIMD', 'OpenMP', 'CUDA', 'CUDA + OpenMP + SIMD'}, 'Location', 'northwest');
title('不同优化方法的执行时间');
grid on;

% 绘制加速比图
subplot(2, 1, 2);
plot(matrix_sizes, simd_speedup, 'o-', 'LineWidth', 2, 'MarkerSize', 6);
hold on;
plot(matrix_sizes, omp_speedup, 's-', 'LineWidth', 2, 'MarkerSize', 6);
plot(matrix_sizes, cuda_speedup, '^-', 'LineWidth', 2, 'MarkerSize', 6);
plot(matrix_sizes, cuda_omp_simd_speedup, 'd-', 'LineWidth', 2, 'MarkerSize', 6);

xlabel('矩阵规模 (N)');
ylabel('加速比 (相对于串行)');
legend({'SIMD', 'OpenMP', 'CUDA', 'CUDA + OpenMP + SIMD'}, 'Location', 'northwest');
title('不同优化方法的加速比');
grid on;
