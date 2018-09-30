import first_steps_with_tensor_flow

first_steps_with_tensor_flow.california_housing_dataframe['rooms_per_person'] = \
    first_steps_with_tensor_flow.california_housing_dataframe['total_rooms'] / \
    first_steps_with_tensor_flow.california_housing_dataframe['population']

calibration_data = first_steps_with_tensor_flow.train_model(
    learning_rate=0.05,
    steps=500,
    batch_size=5,
    input_feature="rooms_per_person"
)

# 绘制预测结果和真实值的散点图，理想状态是一条对角线
first_steps_with_tensor_flow.plt.figure(figsize=(15, 6))
first_steps_with_tensor_flow.plt.subplot(1, 2, 1)
first_steps_with_tensor_flow.plt.scatter(calibration_data['predictions'], calibration_data['targets'])

# 绘制rooms_per_person的直方图，识别离群值
first_steps_with_tensor_flow.plt.subplot(1, 2, 2)
first_steps_with_tensor_flow.plt.show(first_steps_with_tensor_flow.california_housing_dataframe['rooms_per_person'].hist())

# 截取离群值
# rooms_per_person超过5的自动改成5
first_steps_with_tensor_flow.california_housing_dataframe['rooms_per_person'] = \
    first_steps_with_tensor_flow.california_housing_dataframe['rooms_per_person'].apply(lambda x: min(x, 5))

first_steps_with_tensor_flow.plt.show(first_steps_with_tensor_flow.california_housing_dataframe['rooms_per_person'].hist())

# 重新训练
calibration_data = first_steps_with_tensor_flow.train_model(
    learning_rate=0.05,
    steps=500,
    batch_size=5,
    input_feature="rooms_per_person"
)

first_steps_with_tensor_flow.plt.scatter(calibration_data['predictions'], calibration_data['targets'])
first_steps_with_tensor_flow.plt.show()

