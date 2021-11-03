# Intro-to-ML
Here are a few different approaches to classify images using TensorFlow.

deeplearning_models.py contains three different types of models for classifying images (sequential model, functional model, and model inherited from the keras class.
These models are used to classify images from the MNIST (Modified National Institue of Standards and Technology) database and the GTSRB (German Traffic Sign Recognition Benchmark) database.
my_utils.py contains functions used to prepare the images from the databases for the classifying models.
mnist_example.py creates a model that yields >95% accuracy in the identification of MNIST test data and can be further improved by tuning model variables.
street_signs_example.py creates a model that yields >95% accuracy in the identification of GTSRB test data. The best version of the model (highest accuracy) saves in a folder named "Models" and saved within the parent folder.
mypredictor.py takes an image as an input (defined as "img_path") and returns a prediction of the image class using the best model saved in the "Models" folder.

The MNIST dataset is a default dataset within TensorFlow and can be imported directly from TensorFlow.
The GTSRB dataset can be downloaded from Kaggle (https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign).
