import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist         # библиотека базы выборок Mnist здесь мы позволяем ее подключить
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten

# x_train ИЗОБРАЖЕНИЕ цифа обучаюшей выборки
# y_train ВЕКТОР соответствующих значений цифр (напримеб если на i-м изображении нарисованно 5 то у y_train[i] = 5
# x_test ИЗОБРАЖЕНИЕ цифа тестовой выборки
# y_test ВЕКТОР соответствующих значений цифр для тестовой выборки

(x_train, y_train), (x_test, y_test) = mnist.load_data() # загружаем изображение 28 на 28 пикселей

# стандартизация входных данных от 0 до 1 сети плохо работают с болльшими цифрами поэтому их вводим в диапазон от 0 до 1
x_train = x_train / 255
x_test = x_test / 255
# создаем вектор из 10 цифр на выходе а не сколяр как у y_train и y_test например 000100000 вместо 4
y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

# отображение первых 25 изображений из обучающей выборки
plt.figure(figsize=(10,5))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_train[i], cmap=plt.cm.binary)

plt.show()

model = keras.Sequential([ # создаем сеть
    Flatten(input_shape=(28, 28, 1)), # 28*28 = 784 входа по числу пикслей в изображении 1 = байе 1 пиксель в градациях серого (вытягиваем в один вектор длиной 784 пикселя и так подаем в сеть
    Dense(128, activation='relu'),# 128 скрытых слоёв ( сами выбрали количество скрытых нейронов в слое)
    Dense(10, activation='softmax')# 10 выходов
])

print(model.summary())      # вывод структуры НС в консоль

# ----если мы хотим сами оптимизировать то добовляем так

myAdam = keras.optimizers.Adam(learning_rate = 0.1) # дополнительно указываем шаг сходимости learning_rate = 0.1

# если хотим другой отимизатор
# myOpt = keras.optimizer.SGD(learning_rate=0.1, momenztum=0.0, nesterov=True)
#----

#model.compile(optimizer='adam',  # градиентный спуск
model.compile(optimizer = myAdam,  # градиентный спуск
             loss='categorical_crossentropy',
             metrics=['accuracy']) # показатель обучения сети


model.fit(x_train, y_train_cat, batch_size=32, epochs=10, validation_split=0.2) # запскаем процес обучения
#  batch_size=32 после каждых 32 ёизображений мы будем коректировать весовые кофициенты
# validation_split=0.2 будем брать 20% из обучающей выбоки и перемешать в валидацию

model.evaluate(x_test, y_test_cat) # подаем на вход тестовую выборку



# ради интереса подаём на вход одну цифру для опознания из тестовой выборки

n = 1
x = np.expand_dims(x_test[n], axis=0)  # model.predict(x) подрузомевает подачю трёх мерного тензора поэтому добавляем axis=0
res = model.predict(x)
print( res )
print(f"Распознаная цифра = { np.argmax(res)}" ) # берем максимальный индекс

plt.imshow(x_test[n], cmap=plt.cm.binary)
plt.show()

# [[5.5187122e-11 7.0017217e-07 9.9999928e-01 2.1330167e-11 9.1711278e-22
#   3.4918658e-11 9.6178543e-12 2.2419909e-16 5.3975160e-08 2.8452776e-16]]
# Распознаная цифра = 2
# здесь у ДВОЙКИ максималный индекс = 9.9999928e-01 значит на картинке изображена 2



# Распознавание всей тестовой выборки
pred = model.predict(x_test)
pred = np.argmax(pred, axis=1)

print(pred.shape)

print(pred[:20]) # берем 20 ресунков и проверяем их на ошибки
print(y_test[:20])
# (10000,)
# [7 2 1 0 4 1 4 9 5 9 0 6 9 0 1 5 9 7 3 4]
# [7 2 1 0 4 1 4 9 5 9 0 6 9 0 1 5 9 7 3 4]


# выделем все неверные результаты

mask = pred == y_test
print(mask[:10])

x_false = x_test[~mask] # выделем только не верные результаты
y_false = x_test[~mask] #

print(x_false.shape)
# (221, 28, 28) 228 изображений из 10000 определены не верно

# Вывод первых 25 неверных результатов
plt.figure(figsize=(10,5))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_false[i], cmap=plt.cm.binary)

plt.show()


for i in range(5):
    print("Значение сети : "+str(y_false[i]))
    plt.imshow(x_false[i], cmap=plt.cm.binary)
    plt.show()