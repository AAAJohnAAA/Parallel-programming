% 数据：矩阵规模、串行执行时间、并行执行时间（OpenMP）、加速比
matrix_sizes = [32, 64, 128, 256, 512, 1024, 2048];
serial_time = [0.0005, 0.0020, 0.0085, 0.0340, 0.1360, 0.5440, 2.1780];
parallel_time = [0.0003, 0.0010, 0.0040, 0.0150, 0.0600, 0.2400, 0.9800];

% 计算加速比
speedup = serial_time ./ parallel_time;

% 绘制执行时间图
figure;
subplot(2, 1, 1);
plot(matrix_sizes, serial_time, '-o', 'DisplayName', '串行版本', 'LineWidth', 2);
hold on;
plot(matrix_sizes, parallel_time, '-s', 'DisplayName', 'OpenMP并行版本', 'LineWidth', 2);
xlabel('矩阵规模 (N)');
ylabel('执行时间 (秒)');
title('高斯消元法在不同矩阵规模下的执行时间');
legend('show');
grid on;

% 绘制加速比图
subplot(2, 1, 2);
plot(matrix_sizes, speedup, '-^', 'DisplayName', '加速比', 'LineWidth', 2);
xlabel('矩阵规模 (N)');
ylabel('加速比');
title('高斯消元法在不同矩阵规模下的加速比');
legend('show');
grid on;
