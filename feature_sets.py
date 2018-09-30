import first_steps_with_tensor_flow
import validation

# 计算皮尔逊相关系数矩阵
correlation_dataframe = validation.training_examples.copy()
correlation_dataframe['target'] = validation.training_targets['median_house_value']

first_steps_with_tensor_flow.display.display(correlation_dataframe.corr())

# 根据相关矩阵选取特征
minimal_features = [
  "median_income",
  "latitude",
]

minimal_training_examples = validation.training_examples[minimal_features]
minimal_validation_examples = validation.validation_examples[minimal_features]

'''
_ = first_steps_with_tensor_flow.train_model(
    learning_rate=0.01,
    steps=500,
    batch_size=5,
    training_examples=minimal_training_examples,
    training_targets=validation.training_targets,
    validation_examples=minimal_validation_examples,
    validation_targets=validation.validation_targets)

# 绘制latitude-median_house_value图，寻找latitude更明显的特征
first_steps_with_tensor_flow.plt.scatter(first_steps_with_tensor_flow.california_housing_dataframe['latitude'],
                                         first_steps_with_tensor_flow.california_housing_dataframe['median_house_value'])

first_steps_with_tensor_flow.plt.show()
'''

# 分箱：形成32-33, 33-34, ...的纬度
LATITUDE_RANGES = zip(range(32, 44), range(33, 45))


def select_and_transform_features(source_df):
    selected_examples = first_steps_with_tensor_flow.pd.DataFrame()
    selected_examples['median_income'] = source_df['median_income']
    for r in LATITUDE_RANGES:
        selected_examples['latitude_%d_to_%d' % r] = source_df['latitude'].apply(
            lambda l: 1.0 if l >= r[0] and l < r[1] else 0.0)
    return selected_examples


selected_training_examples = select_and_transform_features(validation.training_examples)
selected_validation_examples = select_and_transform_features(validation.validation_examples)

print(selected_training_examples)

_ = first_steps_with_tensor_flow.train_model(
    learning_rate=0.01,
    steps=500,
    batch_size=5,
    training_examples=selected_training_examples,
    training_targets=validation.training_targets,
    validation_examples=selected_validation_examples,
    validation_targets=validation.validation_targets)

