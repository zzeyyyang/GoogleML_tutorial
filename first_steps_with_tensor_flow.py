# 设置
from __future__ import print_function

import math

from IPython import display
from matplotlib import cm
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset


tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

california_housing_dataframe = pd.read_csv("https://dl.google.com/mlcc/mledu-datasets/california_housing_train.csv", sep=",")

california_housing_dataframe = california_housing_dataframe.reindex(
    np.random.permutation(california_housing_dataframe.index))


# 定义输入函数
def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    # 将pandas DataFrame数据转化成np array的字典
    features = {key: np.array(value) for key, value in dict(features).items()}

    # 建立数据集
    # 数据集中每个元素是传入的数据，表示成(features, targets)形式
    ds = Dataset.from_tensor_slices((features, targets))
    ds = ds.batch(batch_size).repeat(num_epochs)

    if shuffle:
        ds = ds.shuffle(10000)

    # 返回下一批数据
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


# 定义特征
def preprocess_features(california_housing_dataframe):
    selected_features = california_housing_dataframe[
        ['latitude',
         'longitude',
         'housing_median_age',
         'total_rooms',
         'total_bedrooms',
         'population',
         'households',
         'median_income']]
    processed_features = selected_features.copy()

    processed_features['rooms_per_person'] = california_housing_dataframe['total_rooms'] / \
                                             california_housing_dataframe['population']

    # 截取离群值（这里仅对rooms_per_person进行）
    processed_features['rooms_per_person'] = processed_features['rooms_per_person'].apply(lambda x: min(x, 4))

    return processed_features


# 定义标签
def preprocess_targets(california_housing_dataframe):
    output_targets = pd.DataFrame()

    output_targets['median_house_value'] = california_housing_dataframe['median_house_value'] / 1000.0

    return output_targets


# 配置特征列
def construct_feature_columns(input_features):
    return set([tf.feature_column.numeric_column(my_feature) for my_feature in input_features])


# 构建并训练模型
def train_model(learning_rate, steps, batch_size, training_examples, training_targets, validation_examples, validation_targets):
    # 设定输出误差轮数
    periods = 10
    steps_per_period = steps / periods

    # 配置线性回归器
    my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)  # 使用SGD（批量随机梯度下降法）训练
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)  # 梯度剪裁：设置梯度上限，防止梯度爆炸
    linear_regressor = tf.estimator.LinearRegressor(feature_columns=construct_feature_columns(training_examples), optimizer=my_optimizer)

    # 配置输入数据
    training_input_fn = lambda: my_input_fn(training_examples, training_targets['median_house_value'], batch_size=batch_size)
    predict_training_input_fn = lambda: my_input_fn(training_examples, training_targets['median_house_value'], num_epochs=1, shuffle=False)
    predict_validation_input_fn = lambda: my_input_fn(validation_examples, validation_targets['median_house_value'], num_epochs=1, shuffle=False)

    # 训练模型
    print('Training model...')
    print('RMSE (on training data): ')
    training_root_mean_squared_errors = []
    validation_root_mean_squared_errors = []
    for period in range(periods):
        linear_regressor.train(input_fn=training_input_fn, steps=steps_per_period)

        # 模型预测
        training_predictions = linear_regressor.predict(input_fn=predict_training_input_fn)
        training_predictions = np.array([item['predictions'][0] for item in training_predictions])

        validation_predictions = linear_regressor.predict(input_fn=predict_validation_input_fn)
        validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])

        # 误差计算
        training_root_mean_squared_error = math.sqrt(metrics.mean_squared_error(training_predictions, training_targets))
        validation_root_mean_squared_error = math.sqrt(metrics.mean_squared_error(validation_predictions, validation_targets))
        print(' period %02d : %02f' % (period, training_root_mean_squared_error))

        training_root_mean_squared_errors.append(training_root_mean_squared_error)
        validation_root_mean_squared_errors.append(validation_root_mean_squared_error)
    print('Model training finished.')

    # 画出误差变化图
    plt.ylabel('RMSE')
    plt.xlabel('Periods')
    plt.title('Root Mean Squared Error vs. Periods')
    plt.tight_layout()
    plt.plot(training_root_mean_squared_errors, label='training')
    plt.plot(validation_root_mean_squared_errors, label="validation")
    plt.legend()
    plt.show()

    return linear_regressor

