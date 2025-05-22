#Step 1: Install Required Libraries:
!pip install tensorflow pillow ipywidgets

#Step 2: Import Libraries:
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
from io import BytesIO

#Step 3: Load and Normalize the MNIST Dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

#Step 4: Build the CNN Model
model = tf.keras.Sequential([
tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
tf.keras.layers.MaxPooling2D((2,2)),
tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
tf.keras.layers.MaxPooling2D((2,2)),
tf.keras.layers.Flatten(),

tf.keras.layers.Dense(64, activation='relu'),
tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

#Step 5: Save and Load the Model
model.save("handwriting_model.h5")
model = tf.keras.models.load_model("handwriting_model.h5")

#Step 6: Preprocess the Uploaded Image
def preprocess_image(image):
image = image.convert('L')
img_array = np.array(image)
if np.mean(img_array) > 127:
img_array = 255 - img_array
img_array = np.where(img_array > 50, 255, 0).astype(np.uint8)
image = Image.fromarray(img_array)
bbox = image.getbbox()
if bbox:
image = image.crop(bbox)
image.thumbnail((20, 20), Image.Resampling.LANCZOS)
new_img = Image.new('L', (28, 28), 0)
new_img.paste(image, ((28 - image.width) // 2, (28 - image.height) // 2))
img_array = np.array(new_img) / 255.0
img_array = img_array.reshape(1, 28, 28, 1)
return img_array

#Step 7: Predict the Handwritten Digit:
def predict_image(image):
processed = preprocess_image(image)
prediction = model.predict(processed)
predicted_digit = np.argmax(prediction)
print(f"Predicted Digit: {predicted_digit}")
plt.imshow(processed[0, :, :, 0], cmap='gray')
plt.title(f'Prediction: {predicted_digit}')
plt.axis('off')
plt.show()

#Step 8: Upload Image Widget for User Input:
upload_widget = widgets.FileUpload(accept='.jpg,.jpeg,.png', multiple=False)
def handle_upload(change):
for filename in upload_widget.value:
content = upload_widget.value[filename]['content']
image = Image.open(BytesIO(content))
predict_image(image)
upload_widget.observe(handle_upload, names='value')
display(upload_widget)
