% 数据：矩阵规模、串行执行时间、并行执行时间（重复创建线程、只创建一次线程）、加速比
matrix_sizes = [32, 64, 128, 256, 512, 1024, 2048];
serial_time = [0.0005, 0.0024, 0.0105, 0.0425, 0.1680, 0.6770, 2.7420];
parallel_repeated_time = [0.0003, 0.0012, 0.0045, 0.0170, 0.0700, 0.2800, 1.1300];
parallel_single_time = [0.0002, 0.0009, 0.0032, 0.0125, 0.0530, 0.2200, 0.8900];

% 计算加速比
speedup_repeated = serial_time ./ parallel_repeated_time;
speedup_single = serial_time ./ parallel_single_time;

% 绘制执行时间图
figure;
subplot(2, 1, 1);
plot(matrix_sizes, serial_time, '-o', 'DisplayName', '串行版本', 'LineWidth', 2);
hold on;
plot(matrix_sizes, parallel_repeated_time, '-s', 'DisplayName', '重复创建线程并行', 'LineWidth', 2);
plot(matrix_sizes, parallel_single_time, '-^', 'DisplayName', '只创建一次线程并行', 'LineWidth', 2);
xlabel('矩阵规模 (N)');
ylabel('执行时间 (秒)');
title('高斯消元法在不同矩阵规模下的执行时间');
legend('show');
grid on;

% 绘制加速比图
subplot(2, 1, 2);
plot(matrix_sizes, speedup_repeated, '-s', 'DisplayName', '重复创建线程并行加速比', 'LineWidth', 2);
hold on;
plot(matrix_sizes, speedup_single, '-^', 'DisplayName', '只创建一次线程并行加速比', 'LineWidth', 2);
xlabel('矩阵规模 (N)');
ylabel('加速比');
title('高斯消元法在不同矩阵规模下的加速比');
legend('show');
grid on;
