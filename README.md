运行本实验：

1. 提取特征（data目录内提取特征维度为64，data256目录内提取特征为256）：
```bash
cd sound_event_detection/data256;
bash prepare_data.sh path_to_your_data
cd ..;
```

2. 训练、测试:
```bash
python run.py train_evaluate configs/256_interpolate_aug.yaml data256/eval/feature.csv data256/eval/label.csv
```

3. 结果在experiments中生成，最优结果已在experiments中给出。

4. 项目报告见Report.pdf
