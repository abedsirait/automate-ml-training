import tensorflow as tf
import pandas as pd

# Baca data dari CSV
data = pd.read_csv('data/train.csv')
features = data[['feature1', 'feature2']]
labels = data['label']

# Membuat dan melatih model sederhana
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(features.shape[1],)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(features, labels, epochs=10)

# Menyimpan model
model.save('model/my_model')

# Menyimpan metrik ke file
with open('metrics.txt', 'w') as f:
    loss, accuracy = model.evaluate(features, labels)
    f.write(f'Loss: {loss}, Accuracy: {accuracy}')
