import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # включили режим работы на CPU  А НЕ НА  GPU

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import Dense

c = np.array([-40, -10, 0, 8, 15, 22, 38])  # подаем на вход
f = np.array([-40, 14, 32, 46, 59, 72, 100])  # что мы хотим получить для обучения сети

model = keras.Sequential() # создаем сеть класа Sequential
model.add(Dense(units=1, input_shape=(1,), activation='linear')) # класс Dence создаёт слой нейронов  units=1
# количество входлв в нашем случае один вход input_shape=(1,) и функция активации  activation='linear' f(x) = x
model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(0.1)) # критерии качества сети и способ градиентного спуска с шагом 0.1

history = model.fit(c, f, epochs=500, verbose=0) # алгоритм обучения нашей сети
# вход, выход, количество эпох,  verbose=0 не будем выводить в терминал служебную инфу во врея обученя сети


plt.plot(history.history['loss'])
plt.grid(True)
plt.show()

print("Обучение завершено")

print(model.predict([100]), "Результат от 100")  # после завершения обучения сети подали на вход 100 чтоб посмотреть как работает сеть
# должно быть чтото такое [[211.33748]]
print(model.get_weights(), "Вывели весовые кофециенты") # вывели весовые кофециенты для сети [array([[1.8205819]], dtype=float32), array([29.27928], dtype=float32)]
# [1.8205819 и 29.27928