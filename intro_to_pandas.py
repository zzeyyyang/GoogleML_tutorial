# -*- coding: utf-8 -*- 

# pandas数据类型类似.csv文件

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Series是一个单一的列
city_name = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])

print(city_name)

population = pd.Series([852469, 1015785, 485199])

print(population)

# DataFrame是一个关系表格，包含多个行和已命名的列
# 一个DataFrame包含一个或多个Series
cities = pd.DataFrame({'City name': city_name, 'Population': population})

# DataFrame.decribe()函数可以显示数据表格的有趣信息
print(cities.describe())

# DataFrame.head()函数可以显示表格的前几行
print(cities.head(1))

# DataFrame.hist()函数可以绘制某列中值分布的直方图
# plt.show(cities.hist('Population'))

# 访问数据
print(type(cities['City name']))
print(cities['City name'])

# 先确定列，后确定行
print(type(cities['City name'][1]))
print(cities['City name'][1])

# 整体按行访问
print(type(cities[:2]))
print(cities[:2])

# 基本运算
print(population / 1000)

# Series可作为多数numpy函数的参数
print(np.log(population))

# Series.apply()函数可以像python一样接受lambda函数，该函数可以作用于每个值
print(population.apply(lambda val: val > 1000000))

# Series数值修改
cities['Area square miles'] = pd.Series([46.87, 176.53, 97.92])
cities['Population density'] = cities['Population'] / cities['Area square miles']

print(cities)

# Training 1
cities['Is wide and has saint name'] = cities['City name'].apply(lambda name: name.startswith('San')) & (cities['Area square miles'] > 50)

print(cities)

# Series和DataFrame对象都有index属性，每个Series项或每个DataFrame行会被赋予一个标志符值
# index值在创建时指定，是稳定的，不随数据排序的改变发生变化
print(city_name.index)
print(cities.index)

# DataFrame.reindex()函数可以手动重排各行顺序
print(cities.reindex([2, 1, 0]))
print(cities.reindex(np.random.permutation(cities.index)))

# 如果reindex输入数组包含原始DataFrame索引值中没有的值，reindex会为此类“丢失的”索引添加新行，并在所有对应列中填充NaN值
print(cities.reindex([0, 4, 5, 2]))
