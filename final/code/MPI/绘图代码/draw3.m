% 矩阵规模
N = [2000, 2000, 4000, 4000, 5000, 5000];

% 并行版本（以便绘图区分）
version = {'p2', 'p3', 'p2', 'p3', 'p2', 'p3'};

% 效率值
efficiency = [0.23, 0.36, 0.14, 0.99, 0.14, 1.02];

% 提取版本编号用于分组绘图
is_p2 = strcmp(version, 'p2');
is_p3 = strcmp(version, 'p3');


figure;

% 绘制 p2.cpp 效率
plot(N(is_p2), efficiency(is_p2), '-o', 'LineWidth', 2, 'DisplayName', 'p2.cpp (MPI+线程+SIMD)');
hold on;

% 绘制 p3.cpp 效率
plot(N(is_p3), efficiency(is_p3), '-s', 'LineWidth', 2, 'DisplayName', 'p3.cpp (MPI非阻塞)');

% 坐标与图例设置
xlabel('矩阵规模 N');
ylabel('并行效率 E');
title('不同矩阵规模下各版本并行效率对比');
legend('Location', 'best');
grid on;
set(gca, 'FontSize', 12);
