import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import image_dataset_from_directory, load_img, img_to_array
from matplotlib.pyplot import imshow
import numpy as np



class ResNetExtension:
    '''
    Class for binary image classification using the weights
    and architecture (except for last layer) from ResNet50
    '''

    def __init__(self, images_path: str) -> None:
        '''
        images_path is expected to have the following structure:

        images_path/
        --test/
        ----class_a/
        ----class_b/
        --train/
        ----class_a/
        ----class_b/
        '''

        self.images_path = images_path

    def build_model(self) -> None:
        self.model = Sequential()

        self.model.add(
            ResNet50(
                include_top = False, # Do note include last layer 
                weights = 'imagenet', # Use imagenet weights
                pooling = 'max', # Do a max pooling to get flat output instead of 4D tensor
                )
            )

        num_of_output_neurons = 1 # Binary classification
        activation = 'sigmoid' # Since we only have one output neuron
        self.model.add(Dense(num_of_output_neurons, activation = activation))

        # Disable training of the first layer which corresponde to the ResNet50 model we're using
        self.model.layers[0].trainable = False

    def compile_model(self) -> None:
        self.model.compile(
            optimizer = 'adam',
            loss = 'binary_crossentropy',
            metrics = 'accuracy'
            )

    def load_images(self) -> None:

        self.train_generator = image_dataset_from_directory(
            directory = os.path.join(self.images_path, 'train'),
            labels='inferred',
            image_size = (224,224)
            )

        self.test_generator = image_dataset_from_directory(
            directory = os.path.join(self.images_path, 'test'),
            labels='inferred',
            image_size = (224,224)
            )

    def fit_model(self) -> None:
        self.fit_history = self.model.fit(
                self.train_generator,
                steps_per_epoch=10,
                epochs = 10,
                validation_data= self.test_generator,
                validation_steps=10,
        )

    def predict_image(self, path: str) -> str:


        image = load_img(path)
        imshow(image)

        input_arr = img_to_array(image)
        input_arr = np.array([input_arr])
        pred = self.model.predict(input_arr)
        predicted_class = self.train_generator.class_names[round(pred[0][0])]

        return predicted_class