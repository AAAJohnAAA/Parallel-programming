% 数据：矩阵规模、普通版本执行时间、NEON版本执行时间
matrix_sizes = [32, 64, 128, 256, 512, 1024, 2048];
serial_time = [0.0008, 0.0025, 0.0100, 0.0400, 0.1600, 0.6400, 2.5600];  % 普通版本执行时间
neon_time = [0.0005, 0.0014, 0.0048, 0.0185, 0.0700, 0.2900, 1.1000];  % NEON加速版本执行时间

% 计算加速比
speedup = serial_time ./ neon_time;

% 绘制执行时间图
figure;
subplot(2, 1, 1);
plot(matrix_sizes, serial_time, '-o', 'DisplayName', '普通版本', 'LineWidth', 2);
hold on;
plot(matrix_sizes, neon_time, '-s', 'DisplayName', 'NEON加速版本', 'LineWidth', 2);
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
title('NEON加速版本与普通版本的加速比');
legend('show');
grid on;
