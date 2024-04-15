import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

MODULE_HANDLE = 'https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b0/feature_vector/2'

IMAGE_SIZE = (224, 224)
NUM_CLASSES = 4

#MODULE_HANDLE = "https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/feature_vector/5"  # Example handle, you can replace it

def create_model():
    feature_extractor = hub.KerasLayer(MODULE_HANDLE,
                                       input_shape=(224, 224, 3),  # Input shape of images (height, width, channels)
                                       trainable=False)  # Freeze the weights of the pre-trained model
    
    model = models.Sequential([
        feature_extractor,
        layers.Dense(NUM_CLASSES, activation='softmax')  # Output layer with softmax activation for classification
    ])
    
    return model

# Create an instance of the model
model = create_model()

train_dir = 'training/dataset/train'
test_dir = 'training/dataset/test'
# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMAGE_SIZE,
    batch_size=32,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMAGE_SIZE,
    batch_size=32,
    class_mode='categorical')


history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples/train_generator.batch_size,
    epochs=1,  # You can adjust the number of epochs
    validation_data=validation_generator,
    validation_steps=validation_generator.samples/validation_generator.batch_size)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(validation_generator)
print('Test Accuracy:', test_accuracy)
