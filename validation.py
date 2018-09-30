import first_steps_with_tensor_flow

# 划分训练集
training_examples = first_steps_with_tensor_flow.preprocess_features(first_steps_with_tensor_flow.california_housing_dataframe.head(12000))
# first_steps_with_tensor_flow.display.display(training_examples.describe())

training_targets = first_steps_with_tensor_flow.preprocess_targets(first_steps_with_tensor_flow.california_housing_dataframe.head(12000))
# first_steps_with_tensor_flow.display.display(training_targets.describe())

# 划分验证集
validation_examples = first_steps_with_tensor_flow.preprocess_features(first_steps_with_tensor_flow.california_housing_dataframe.tail(5000))
# first_steps_with_tensor_flow.display.display(validation_examples.describe())

validation_targets = first_steps_with_tensor_flow.preprocess_targets(first_steps_with_tensor_flow.california_housing_dataframe.tail(5000))
# first_steps_with_tensor_flow.display.display(validation_targets.describe())

'''
# 绘制经纬度与房屋价值中位数曲线图
first_steps_with_tensor_flow.plt.figure(figsize=(13, 8))

ax = first_steps_with_tensor_flow.plt.subplot(1, 2, 1)
ax.set_title('Validation Date')

ax.set_autoscaley_on(False)
ax.set_ylim([32, 43])
ax.set_autoscalex_on(False)
ax.set_xlim([-126, -112])
first_steps_with_tensor_flow.plt.scatter(validation_examples['longitude'],
                                         validation_examples['latitude'],
                                         cmap='coolwarm',
                                         c=validation_targets['median_house_value'] / validation_targets["median_house_value"].max())

ax = first_steps_with_tensor_flow.plt.subplot(1, 2, 2)
ax.set_title('Training Data')

ax.set_autoscaley_on(False)
ax.set_ylim([32, 43])
ax.set_autoscalex_on(False)
ax.set_xlim([-126, -112])
first_steps_with_tensor_flow.plt.scatter(training_examples['longitude'],
                                         training_examples['latitude'],
                                         cmap='coolwarm',
                                         c=training_targets['median_house_value'] / training_targets["median_house_value"].max())

first_steps_with_tensor_flow.plt.show()

# 训练集训练并验证集验证
linear_regressor = first_steps_with_tensor_flow.train_model(
    learning_rate=0.00003,
    steps=500,
    batch_size=5,
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)

# 加载测试集
california_housing_test_data = first_steps_with_tensor_flow.pd.read_csv("https://dl.google.com/mlcc/mledu-datasets/california_housing_test.csv", sep=",")

test_examples = first_steps_with_tensor_flow.preprocess_features(california_housing_test_data)
test_targets = first_steps_with_tensor_flow.preprocess_targets(california_housing_test_data)

predict_test_input_fn = lambda: first_steps_with_tensor_flow.my_input_fn(test_examples, test_targets['median_house_value'], num_epochs=1, shuffle=False)

test_predictions = linear_regressor.predict(input_fn=predict_test_input_fn)
test_predictions = first_steps_with_tensor_flow.np.array([item['predictions'][0] for item in test_predictions])

root_mean_squared_error = first_steps_with_tensor_flow.math.sqrt(
    first_steps_with_tensor_flow.metrics.mean_squared_error(test_predictions, test_targets))

print("Final RMSE (on test data): %0.2f" % root_mean_squared_error)
'''
