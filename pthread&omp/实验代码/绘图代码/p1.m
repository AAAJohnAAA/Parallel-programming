% 数据
matrix_sizes = [32, 64, 128, 256, 512, 1024, 2048];
serial_times = [0.0005, 0.0012, 0.0045, 0.0180, 0.0720, 0.2900, 1.1500];
repeated_parallel_times = [0.0003, 0.0007, 0.0025, 0.0100, 0.0380, 0.1450, 0.5700];
single_thread_parallel_times = [0.00025, 0.0006, 0.0020, 0.0085, 0.0335, 0.1200, 0.4800];

% 绘图
figure;

% 绘制串行版本
plot(matrix_sizes, serial_times, '-o', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', '串行版本');

% 绘制重复创建线程并行版本
hold on;
plot(matrix_sizes, repeated_parallel_times, '-s', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', '重复创建线程并行版本');

% 绘制只创建一次线程并行版本
plot(matrix_sizes, single_thread_parallel_times, '-^', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', '只创建一次线程并行版本');

% 设置图形属性
xlabel('矩阵规模', 'FontSize', 12);
ylabel('执行时间 (秒)', 'FontSize', 12);
title('ARM平台下不同矩阵规模的高斯消元法执行时间', 'FontSize', 14);
legend('show', 'FontSize', 12);
grid on;

% 设置横坐标为32，64，128等距标识
xticks(matrix_sizes);
xticklabels(arrayfun(@(x) num2str(x), matrix_sizes, 'UniformOutput', false));

% 设置纵坐标为对数坐标
set(gca, 'YScale', 'log'); 
