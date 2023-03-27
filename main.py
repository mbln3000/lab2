#1) Есть два файла с данными турагенства: email.csv и username.csv.
# C ними нужно проделать все манипуляции, указанные в лекции 2, а именно:
#a)	Группировка и агрегирование ( сгруппировать набор данных по значениям в столбце,
# а затем вычислить среднее значение для каждой группы)

import pandas as pd
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import pylab


df = pd.read_csv('email.csv')
grouped_data = df.groupby('Trip count').mean()
print(grouped_data)
print(grouped_data.mean())

# b)	Обработка отсутствующих данных (заполнение отсутствующих значений
# определенным значением или интерполяция отсутствующих значений)

df = pd.read_csv('username.csv')
df = df.fillna(value=0)
print(df)

# c)	Слияние и объединение данных (соединить два DataFrames в определенном столбце)

df1 = pd.read_csv('username.csv')
df2 = pd.read_csv('email.csv')
merged_data = pd.merge(df1, df2, on='Trip count')
print(merged_data)


# 2) Преобразование данных (pivot):
# a) Нужно создать сводную таблицу так, чтобы в index были
# столбцы “Rep”, “Manager” и “Product”, а в values “Price” и “Quantity”.
# Также нужно использовать функцию aggfunc=[numpy.sum]
# и заполнить отсутствующие значения нулями.
# В итоге можно будет увидеть количество проданных продуктов и их стоимость,
# отсортированные по имени менеджеров и директоров.

df = pd.read_csv('sales.csv')
df = df.fillna(value=0)
print(pd.pivot_table(df, index=["Rep", "Manager", "Product"], values=["Price", "Quantity"], aggfunc=np.sum))

# b) Учебный файл (data.csv) + практика Dataframe.pivot.
# Поворот фрейма данных и суммирование повторяющихся значений.

df = pd.read_csv('data.csv')
print(pd.pivot_table(df, index=["Date"], columns='Product', values=["Sales"], aggfunc='sum'))

# 3) Визуализация данных (можно использовать любой из учебных csv-файлов).
# a) Необходимо создать простой линейный график из файла csv (два любых столбца, в которых есть зависимость)

df = pd.read_csv('email.csv')
x = df['Total price']
y = df['Trip count']
fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_xlabel('Total price')
ax.set_ylabel('Trip count')
ax.set_title('Simple Line Plot')
plt.show()

# b) Создание визуализации распределения набора данных.
# Создать произвольный датасет через np.random.normal
# или использовать датасет из csv-файлов, потом построить гистограмму.

data = np.random.normal(60, 20, 200)
plt.hist(data, bins=20)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of Data')
plt.show()

# c) Сравнение нескольких наборов данных на одном графике.
# Создать два набора данных с нормальным распределением или использовать данные из файлов.
# Оба датасета отразить на одной оси, добавить легенду и название.

data1 = np.random.normal(40, 30, 200)
data2 = np.random.normal(60, 15, 200)
fig, ax = plt.subplots()
ax.plot(data1, label='Dataset 1')
ax.plot(data2, label='Dataset 2')
ax.set_xlabel('Index')
ax.set_ylabel('Value')
ax.set_title('Comparison of Two Datasets')
ax.legend()
plt.show()

# d) Построение математической функции.
# Создать данные для x и y (где x это numpy.linspace,
# а y - заданная в условии варианта математическая функция).
# Добавить легенду и название графика.
# Вариант 1 - функция sin
# Вариант 2 - функция cos

x = np.linspace(-np.pi, np.pi, 100)
y = np.cos(x)
fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_title('Plot of the Cos Function')
plt.show()

#e) Моделирование простой анимации. Создать данные для x и y (где x это numpy.linspace,
# а y - математическая функция). Запустить объект line,
# ввести функцию animate(i) c методом line.set_ydata()
# и создать анимированный объект FuncAnimation.
# a)	Шаг 1: смоделировать график sin(x) (или cos(x)) в движении.
# b)	Шаг 2: добавить к нему график cos(x) (или sin(x)) так,
# чтобы движение шло одновременно и оба графика отображались на одной оси.

x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x)
fig, ax = plt.subplots()
line, = ax.plot(x, y)


def animate(i):
    line.set_ydata(np.sin(x + i / 10))
    line2.set_ydata(np.cos(x + i / 10))
    return line, line2,

z = np.cos(x)
line2, = ax.plot(x, z)

ani = animation.FuncAnimation(fig, animate, frames=100, blit=True)
plt.show()

# 4) Загрузка CSV-файла в DataFrame. Используя pandas, напишите скрипт,
# который загружает CSV-файл в DataFrame и отображает первые 5 строк df.

df = pd.read_csv('cars.csv')
print(df.head())

# 5) Выбор столбцов из DataFrame.
# a)	Используя pandas, напишите сценарий, который из DataFrame файла sales.csv выбирает только те строки,
# в которых Status = presented, и сортирует их по цене от меньшего к большему.

df = pd.read_csv('sales.csv')
df = df.loc[df['Status'] == 'presented']
df.sort_values(by=['Price'], inplace=True)
print(df)

# b)	Из файла climate.csv отображает в виде двух столбцов названия и коды (rw_country_code) тех стран,
# у которых cri_score больше 100, а fatalities_total не более 10.

df = pd.read_csv('climate.csv')
df = df.loc[df['cri_score'] > 100]
df = df.loc[df['fatalities_total'] < 10]
new_df = df[["country", "rw_country_code"]]

print(new_df)

# c)	Из файла cars.csv отображает названия 50 первых американских машин,
# у которых расход топлива MPG не менее 25, а частное внутреннего объема (Displacement)
# и количества цилиндров не более 40. Названия машин нужно вывести в алфавитном порядке.

df = pd.read_csv('cars.csv')
df = df.loc[df['Origin'] == 'US']
df = df.loc[df['MPG'] >= 25]
# df = df.loc[df['Displacement'] <= 40] #этому условию ничего не соответсвует
df = df.loc[df['Cylinders'] <= 40]
df.sort_values(by=['Car'], inplace=True)
print(df.head(50))

# 6) Вычисление статистики для массива numpy
# Используя numpy, напишите скрипт, который загружает файл CSV
# в массив numpy и вычисляет среднее значение, стандартное отклонение
# и максимальное значение массива. Для тренировки используйте файл data.csv,
# а потом любой другой csv-файл от 20 строк.

data = np.genfromtxt('data.csv', delimiter=',')
mean = np.mean(data)
stand = np.std(data)
max_value = np.max(data)
print(f"Среднее значение: {mean}")
print(f"Стандартное отклонение: {stand}")
print(f"Макс. значение: {max_value}")

data = np.genfromtxt('random.csv', delimiter=';')
mean = np.mean(data)
stand = np.std(data)
max_value = np.max(data)
print(f"Среднее значение: {mean}")
print(f"Стандартное отклонение: {stand}")
print(f"Макс. значение: {max_value}")


# 7) Операции с матрицами: Используя numpy, напишите сценарий,
# который создает матрицу и выполняет основные математические операции,
# такие как сложение, вычитание, умножение и транспонирование матрицы.

matrix1 = np.array([[9,8],[4,5]])
matrix2 = np.array([[1,2],[15,16]])

add_matrix = np.add(matrix1, matrix2)
print(f"Сложение: {add_matrix}")
sub_matrix = np.subtract(matrix1, matrix2)
print(f"Вычитание: {sub_matrix}")
mul_matrix = matrix1.dot(matrix2)
print(f"Умножение: {mul_matrix}")

matrix3 = np.array([[0, 1, 2], [4, 5, 6]])
transp = matrix3.transpose()
print(f"Транспонирование: {transp}")