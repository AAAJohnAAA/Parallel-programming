% 数据：矩阵规模、串行执行时间、并行执行时间、加速比
matrix_sizes = [32, 64, 128, 256, 512, 1024, 2048];
serial_time = [0.0005, 0.0012, 0.0045, 0.0180, 0.0720, 0.2900, 1.1500];
parallel_time = [0.0003, 0.0007, 0.0025, 0.0100, 0.0380, 0.1450, 0.5700];
speedup = serial_time ./ parallel_time;

% 绘制执行时间图
figure;
subplot(2, 1, 1);
plot(matrix_sizes, serial_time, '-o', 'DisplayName', '串行版本', 'LineWidth', 2);
hold on;
plot(matrix_sizes, parallel_time, '-s', 'DisplayName', '并行版本 (OpenMP)', 'LineWidth', 2);
xlabel('矩阵规模 (N)');
ylabel('执行时间 (秒)');
title('高斯消元法在ARM平台下的执行时间');
legend('show');
grid on;

% 绘制加速比图
subplot(2, 1, 2);
plot(matrix_sizes, speedup, '-^', 'DisplayName', '加速比', 'LineWidth', 2);
xlabel('矩阵规模 (N)');
ylabel('加速比');
title('高斯消元法在ARM平台下的加速比');
legend('show');
grid on;
