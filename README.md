# TransferLearningResNet50

This project is a easy to use binary classifier, that has a very good performance for using the weight and architecture of the ResNet50 CNN. Only the last layer of the model will be taken off and substituted according to generate a binary output.

## Files
- **resnet_transfer_learning.py**\
  File that contains the implementation of the model, getting the weights and architecture os ResNet50 model from keras and making the necessary adjustments to be able to build and train the model. Also has the fuctionality of loading the data from the directory especified.
- **main.ipynb**\
  Example of implementation using the cat vs dog dataset.
  
## How to use
1. Import the class and instantiate it, passing the directory of the data as parameter\
OBS: Directory has to have the following structure:\
dir/\
\-\-train/\
\-\-\-\-class_a/\
\-\-\-\-class_b/\
\-\-test/\
\-\-\-\-class_a/\
\-\-\-\-class_b/

```python
from resnet_transfer_learning import ResNetExtension

# ...
# dir = ...
#

res_ext = ResNetExtension(dir)
```

2. Run method to load the data

```python
res_ext.load_images()
```

3. Run method to build model

```python
res_ext.build_model()
```

4. Compile the model


```python
res_ext.compile_model()
```

5. Fit the model


```python
res_ext.fit_model()
```

6. To make a prediction in a specific image, use the *predict_image* method, simply by passing the image path as argument:

```python
# ...
# test_img_path = ...
# ...
res_ext.predict_image(test_img_path)
```
This method will display the image and and return the predicted class.
