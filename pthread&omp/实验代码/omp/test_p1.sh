#!/bin/bash
MATRIX_SIZES=(32 64 128 256 512 1024 2048)
echo "编译 OpenMP 高斯消元法..."
gcc -O3 -fopenmp -o p1 p1.c
LOG_FILE="p1_test_results.log"
echo "测试结果记录在 $LOG_FILE 中"
echo "矩阵规模 (N), 执行时间 (秒)" > $LOG_FILE
for N in "${MATRIX_SIZES[@]}"; do
    echo "正在测试矩阵规模 N = $N ..."
    START_TIME=$(date +%s.%N)
    ./p1
    END_TIME=$(date +%s.%N)
    EXEC_TIME=$(echo "$END_TIME - $START_TIME" | bc)
    echo "$N, $EXEC_TIME" >> $LOG_FILE
    echo "矩阵规模 N = $N 完成，执行时间：$EXEC_TIME 秒"
done
echo "测试完成，结果已保存到 $LOG_FILE"
