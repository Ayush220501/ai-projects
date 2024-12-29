# -*- coding: utf-8 -*-
"""end to end dog vision.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Jua0KAajosmDpulQTC5DIGZFQsKyv9Et
"""

!pip install tensorflow==2.15.0 tensorflow-hub==0.16.1 tf_keras==2.15.1

"""# 🐶 End-to-end Multil-class Dog Breed Classification
This notebook builds an end-to-end multi-class image classifier using TensorFlow 2.x and TensorFlow Hub.

1. Problem
Identifying the breed of a dog given an image of a dog.

When I'm sitting at the cafe and I take a photo of a dog, I want to know what breed of dog it is.

2. Data
The data we're using is from Kaggle's dog breed identification competition.

https://www.kaggle.com/c/dog-breed-identification/data

3. Evaluation
The evaluation is a file with prediction probabilities for each dog breed of each test image.

https://www.kaggle.com/c/dog-breed-identification/overview/evaluation

4. Features
Some information about the data:

We're dealing with images (unstructured data) so it's probably best we use deep learning/transfer learning.
There are 120 breeds of dogs (this means there are 120 different classes).
There are around 10,000+ images in the training set (these images have labels).
There are around 10,000+ images in the test set (these images have no labels, because we'll want to predict them)
"""

!unzip -o "drive/MyDrive/dog-breed-identification.zip"

# Import nescessary tools
import tensorflow as tf
import tensorflow_hub as hub
print("TF version",tf.__version__)
print("TF Hub version",hub.__version__)

#Check for GPU
print("GPU","available(yess)" if tf.config.list_physical_devices("GPU") else "not available")

"""#Getting our data ready (turning into Tensors)
With all machine learning models, our data has to be in numerical format. So that's what we'll be doing first. Turning our images into Tensors (numerical representations).

Let's start by accessing our data and checking out the labels.
"""

#check the labels of the data
import pandas as pd
labels_csv=pd.read_csv("/content/labels.csv")
print(labels_csv.describe())
print(labels_csv.head())

labels_csv.head()

# How many images are there for each breed?
labels_csv["breed"].value_counts()

labels_csv["breed"].value_counts().plot.bar(figsize=(20,10))

labels_csv["breed"].value_counts().median()

#Lets view an Image
from IPython.display import Image
Image("/content/train/00214f311d5d2247d5dfe4fe24b2303d.jpg")

"""#Getting images and their labels
Let's get a list of all of our image file pathnames.
"""

#Create a pathname from image ID's
filenames=[]

for fname in labels_csv["id"]:
  path="/content/train/"+fname+".jpg"
  filenames.append(path)

#check the first 10
filenames[:10]

#Check whether the number of filenames matches the number of actual images files
import os
if len(os.listdir("/content/train/"))==len(filenames):
  print("Yup! Filenames is matched")
else:
  print("Filenames are not matched")

#One more check
Image(filenames[2850])

labels_csv["breed"][2850]

labels=labels_csv["breed"]
labels

#Turning this into numpy
import numpy as np
labels=labels_csv["breed"].to_numpy()
labels

len(labels)

#See if number of labels matches the number of file names
if len(labels) == len(filenames):
  print("It Matches")
else:
  print("It does not matches")

#Find the unique number values
unique_breeds=np.unique(labels)
unique_breeds

#Turn a single label in to array of booleans
print(labels[0])
labels[0]==unique_breeds

#Turn every label into a boolean value
boolean_labels=[label== unique_breeds for label in labels]
boolean_labels[:5]

#Turning boolean arrays into integers
print(labels[0]) #original label
print(np.where(unique_breeds==labels[0])) # index where label occurs
print(boolean_labels[0].argmax()) # Index where label occurs in boolean array
print(boolean_labels[0].astype(int)) # there will be 1 where sample label occurs

filenames[:20]

"""#Creating our own validation set
Since the dataset from Kaggle doesn't come with a validation set, we're going to create our own.
"""

#Setup X and Y variables
X=filenames
y=boolean_labels

#Set number of images to usefor experimenting
NUM_IMAGES=1000 #@param {type:"slider",min:1000,max:10000,step:1000}

#Importing Libraries
from sklearn.model_selection import train_test_split

#Results are Reproducible
np.random.seed(42)
#Split into training and validation sets
X_train,X_val,y_train,y_val= train_test_split(X[:NUM_IMAGES],y[:NUM_IMAGES],test_size=0.2)

len(X_train),len(X_val),len(y_train),len(y_val)

#Let's have a geez on our dataset
X_train[:5],y_train[:2]

"""#Preprocessing Images (turning images into Tensors)
To preprocess our images into Tensors we're going to write a function which does a few things:

1. Take an image filepath as input
2. Use TensorFlow to read the file and save it to a variable, image
3. Turn our image (a jpg) into Tensors
4. Normalize our image (convert color channel values from from 0-255 to 0-1).
5. Resize the image to be a shape of (224, 224)
6. Return the modified image
7. Before we do, let's see what importing an image looks like.
"""

#Convert an image to Numpy Array
from matplotlib.pyplot import imread
image=imread(filenames[42])
image.shape

image

#Turning image into Tensors
tf.constant(image)[:2]

"""# Now we've seen what an image looks like as a Tensor, let's make a function to preprocess them.

We'll create a function to:

1. Take an image filepath as input
2. Use TensorFlow to read the file and save it to a variable, image
3. Turn our image (a jpg) into Tensors
4. Normalize our image (convert color channel values from from 0-255 to 0-1).
5. Resize the image to be a shape of (224, 224)
6. Return the modified image

More information on loading images in TensorFlow can be seen here: https://www.tensorflow.org/tutorials/load_data/images
"""

# Define image size
IMG_SIZE = 224

def process_image(image_path):
  """
  Takes an image file path and turns it into a Tensor.
  """
  # Read in image file
  image = tf.io.read_file(image_path)
  # Turn the jpeg image into numerical Tensor with 3 colour channels (Red, Green, Blue)
  image = tf.image.decode_jpeg(image, channels=3)
  # Convert the colour channel values from 0-225 values to 0-1 values
  image = tf.image.convert_image_dtype(image, tf.float32)
  # Resize the image to our desired size (224, 244)
  image = tf.image.resize(image, size=[IMG_SIZE, IMG_SIZE])
  return image

"""# Turning our data into batches
Why turn our data into batches?

Let's say you're trying to process 10,000+ images in one go... they all might not fit into memory.

So that's why we do about 32 (this is the batch size) images at a time (you can manually adjust the batch size if need be).

In order to use TensorFlow effectively, we need our data in the form of Tensor tuples which look like this: (image, label).
"""

#Create a simple function to return a tuple (image, label).

def get_image_label(image_path,label):
  '''
  Takes an image path file name and the associated label,
  preprocess the image and returns a tuple of (image,label)

  '''

  image = process_image(image_path)

  return image,label

#Demo of the above code
image=process_image(X[42]),tf.constant(y[42])
image

# Define the batch size, 32 is a good default
BATCH_SIZE = 32

# Create a function to turn data into batches
def create_data_batches(x, y=None, batch_size=BATCH_SIZE, valid_data=False, test_data=False):
  """
  Creates batches of data out of image (x) and label (y) pairs.
  Shuffles the data if it's training data but doesn't shuffle it if it's validation data.
  Also accepts test data as input (no labels).
  """
  # If the data is a test dataset, we probably don't have labels
  if test_data:
    print("Creating test data batches...")
    data = tf.data.Dataset.from_tensor_slices((tf.constant(x))) # only filepaths
    data_batch = data.map(process_image).batch(BATCH_SIZE)
    return data_batch

  # If the data if a valid dataset, we don't need to shuffle it
  elif valid_data:
    print("Creating validation data batches...")
    data = tf.data.Dataset.from_tensor_slices((tf.constant(x), # filepaths
                                               tf.constant(y))) # labels
    data_batch = data.map(get_image_label).batch(BATCH_SIZE)
    return data_batch

  else:
    # If the data is a training dataset, we shuffle it
    print("Creating training data batches...")
    # Turn filepaths and labels into Tensors
    data = tf.data.Dataset.from_tensor_slices((tf.constant(x), # filepaths
                                              tf.constant(y))) # labels

    # Shuffling pathnames and labels before mapping image processor function is faster than shuffling images
    data = data.shuffle(buffer_size=len(x))

    # Create (image, label) tuples (this also turns the image path into a preprocessed image)
    data = data.map(get_image_label)

    # Turn the data into batches
    data_batch = data.batch(BATCH_SIZE)
  return data_batch

# Create training and validation data batches
train_data = create_data_batches(X_train, y_train)
val_data = create_data_batches(X_val, y_val, valid_data=True)

# Check out the different attributes of our data batches
train_data.element_spec, val_data.element_spec

"""#Visualizing Data Batches
Our data is now in batches, however, these can be a little hard to understand/comprehend, let's visualize them!
"""

import matplotlib.pyplot as plt

# Create a function for viewing images in a data batch
def show_25_images(images, labels):
  """
  Displays 25 images from a data batch.
  """
  # Setup the figure
  plt.figure(figsize=(10, 10))
  # Loop through 25 (for displaying 25 images)
  for i in range(25):
    # Create subplots (5 rows, 5 columns)
    ax = plt.subplot(5, 5, i+1)
    # Display an image
    plt.imshow(images[i])
    # Add the image label as the title
    plt.title(unique_breeds[labels[i].argmax()])
    # Turn gird lines off
    plt.axis("off")

train_data

val_data

# Visualize training images from the training data batch
train_images, train_labels = next(train_data.as_numpy_iterator())
show_25_images(train_images, train_labels)

#Now let's visualize in a validation batch
val_images,val_labels=next(val_data.as_numpy_iterator())
show_25_images(val_images,val_labels)

"""# Building a model
Before we build a model, there are a few things we need to define:

The input shape (our images shape, in the form of Tensors) to our model.
The output shape (image labels, in the form of Tensors) of our model.
The URL of the model we want to use from TensorFlow Hub - https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/4
"""

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers, models

#Setup input shape of our model
INPUT_SHAPE=[None ,IMG_SIZE,IMG_SIZE,3] #batch, height, width, color channels

#Setup output shape of our model
OUTPUT_SHAPE=len(unique_breeds)

#Setup model URL from tensorflow hub
MODEL_URL="https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/4"

"""#Now we've got our inputs, outputs and model ready to go. Let's put them together into a Keras deep learning model!

Knowing this, let's create a function which:

1. Takes the input shape, output shape and the model we've chosen as parameters.
2. Defines the layers in a Keras model in sequential fashion (do this first, then this, then that).
3. Compiles the model (says it should be evaluated and improved).
4. Builds the model (tells the model the input shape it'll be getting).
5. Returns the model.

"""

# Create a function which builds a Keras model
def create_model(input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE, model_url=MODEL_URL):
  print("Building model with:", MODEL_URL)

  # Setup the model layers
  model = tf.keras.Sequential([
    hub.KerasLayer(MODEL_URL), # Layer 1 (input layer)
    tf.keras.layers.Dense(units=OUTPUT_SHAPE,
                          activation="softmax") # Layer 2 (output layer)
  ])

  # Compile the model
  model.compile(
      loss=tf.keras.losses.CategoricalCrossentropy(),
      optimizer=tf.keras.optimizers.Adam(),
      metrics=["accuracy"]
  )

  # Build the model
  model.build(INPUT_SHAPE)

  return model

# Create a model and check its details
model = create_model()
model.summary()

"""#Creating callbacks
Callbacks are helper functions a model can use during training to do such things as save its progress, check its progress or stop training early if a model stops improving.

We'll create two callbacks, one for TensorBoard which helps track our models progress and another for early stopping which prevents our model from training for too long.

#TensorBoard Callback
To setup a TensorBoard callback, we need to do 3 things:

Load the TensorBoard notebook extension ✅
Create a TensorBoard callback which is able to save logs to a directory and pass it to our model's fit() function. ✅
Visualize our models training logs with the %tensorboard magic function (we'll do this after model training).
"""

# Commented out IPython magic to ensure Python compatibility.
# Load the TensorBoard notebook extension
# %load_ext tensorboard

"""Make sure to make **logs** before running the further cell"""

import datetime

# Create a function to build a TensorBoard callback
def create_tensorboard_callback():
  # Create a log directory for storing TensorBoard logs
  logdir = os.path.join("/content/logs",
                        # Make it so the logs get tracked whenever we run an experiment
                        datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
  return tf.keras.callbacks.TensorBoard(logdir)

"""#Early Stopping Callback
Early stopping helps stop our model from overfitting by stopping training if a certain evaluation metric stops improving.

https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping
"""

# Create early stopping (once our model stops improving, stop training)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy",
                                                  patience=3) # stops after 3 rounds of no improvements

"""# Training a model(on subnet of data)

Our first model is going to train on 1000 images, to make sure everything is working.
"""

NUM_EPOCHS= 100 #@param {type:"slider", min:10, max:100, step:10}

#check if we are still running on a GPU
print("GPU","available (yes !!!!!!!!!!)"if tf.config.list_physical_devices("GPU") else "not available")

"""# Let's create a function which trains a Model.
* Create a model using 'create_model()'.
* Setup a TensorBoard Callback using 'create_tensorboard_callback()'.
* Call the 'fit()' function on our model passing it to our training data, validation data, number of epochs to train for (NUM_EPOCHS) and the callbacks we would like to use.
* Return the Model
"""

# Build a function to train and returned a train model
def train_model():

  """
  Trains a given model and returns the trained version.
  """

  #Create a model
  model=create_model()

  #Create a Tensorboard session everytime we train a model
  tensorboard=create_tensorboard_callback()

  #Fit the model to the data passing it the callbacks it served
  model.fit(x=train_data,
            epochs=NUM_EPOCHS,
            validation_data=val_data,
            validation_freq=1,
            callbacks=[tensorboard,early_stopping])

  #Return the fitted model
  return model

#Fit the model to the data
model=train_model()

"""**Question** : It looks like our model is overfitting because it performing far better on the training dataset than the validation dataset, what are some ways to prevent model overfitting in deep learning neural networks.

**Note**: Overfitting to begin with is a good thing! it means our model is learning!!!

# Checking the TensorBoard logs
The TensorBoard magic function **(%tensorboard)** will access the logs directory and visualize its contents.
"""

# Commented out IPython magic to ensure Python compatibility.
# %tensorboard --logdir /content/logs

"""# Making and evaluating predictions using trained model."""

val_data

# Making Predictions on the validation data (not used to be train on)
predictions=model.predict(val_data,verbose=1) #Verbose means show me your progress
predictions

predictions[0]

predictions.shape

len(y_val),len(unique_breeds)

#First Predictions
index=1
print(predictions[index])
print(f"Max value (probability of predictions):{predictions[index].max()}")
print(f"Sum:{predictions[index].sum()}")
print(f"Max index: {predictions[index].argmax()}")
print(f"Predicted label:{unique_breeds[predictions[index].argmax()]}")

"""Having the above functionality is great but we want to able to do it at scale
And it would better if we could see the image the predictions is being made on.

**Note**: Predictions Probabilities are also known as confidence intervals
"""

#Turn Prediction probablities into thier respective label(which is easier to understand)

def get_pred_label(prediction_probabilities):
  """
Turns an array of prediction probabilities into a label

"""

  return unique_breeds[np.argmax(prediction_probabilities)]

#Get a predicted label based on array of predicted probabilities
pred_label=get_pred_label(predictions[22])
pred_label

"""Now since our validation data is still in batch dataset, we'll have to unbatchify it to make predictions on the validation images and then compare those predictions to the validation labels (truth labels)."""

val_data

#Create a function to unbatch a batch dataset
def unbatchify(data):

  '''
Take a batched dataset of (image,label) and returns seperate arrays of images and labels.

  '''

  images=[]
  labels=[]

  #Loop through unbatched data
  for image,label in data.unbatch().as_numpy_iterator():
   images.append(image)
   labels.append(unique_breeds[np.argmax(label)])

  return images,labels

#Unbatchify the validation data
val_images,val_labels=unbatchify(val_data)
val_images[0],val_labels[0]

"""Now we have got the ways to get:
* Prediction Labels
* Validation Labels (truth labels)
* Validation Images

Now lets develop a function to make it more visualize.

we'll create a function which:
* Takes an array of predicition probabilities, an array of truth labels and an array of images and integer.
* Convert the prediction probabilities in to a predicted label.
* Plot the predicted label, its predicted probablility and target image on a single plot.
"""

def plot_pred(prediction_probabilities,labels,images,n=1):
  '''
  View the prediction, ground truth and image for sample n

  '''

  pred_probe,true_label,image= prediction_probabilities[n],labels[n],images[n]

  #Get the Pred Label
  pred_label=get_pred_label(pred_probe)

  #Plot images and remove ticks
  plt.imshow(image)
  plt.xticks([])
  plt.yticks([])

  #Applying color if the predictions is true it is change to green or red
  if pred_label==true_label:
    color="green"
  else:
    color="red"

  #Change the plot title to be predicted, probabiloty of prediction and truth label
  plt.title("{} {:2.0f}% {}".format(pred_label,
                                    np.max(pred_probe)*100,
                                    true_label),
                                    color=color)

plot_pred(prediction_probabilities=predictions,
          labels=val_labels,
          images=val_images,
          n=77)

"""Now we have got one function to visualize our models top predictions. Let's make others to view top 10 predictons.

This function will:
* Take an input of predicted probabilities array and a ground truth array and a label.

* Find the prediction using get_pred_label()

* Find the top 10:
      1. Prediction Probability Indexes
      2. Prediction Probability Values
      3. Prediction Labels


* Plot the top 10 prediction probability values and labels, coloring the true greenth label.




"""

def plot_pred_conf(prediction_probabilities, labels,n=1):

  '''
  Plus the top 10 highest prediction confidences along with the truth label for sample n.

  '''

  pred_prob,true_label=prediction_probabilities[n],labels[n]

  #Get the predicted Labels
  pred_label=get_pred_label(pred_prob)

  #Find the top 10 prediction confidence intervals
  top_10_pred_indexes=pred_prob.argsort()[-10:][::-1]

  #Find the top 10 prediction confidence values
  top_10_pred_values=pred_prob[top_10_pred_indexes]

  #Find the top 10 prediction labels
  top_10_pred_labels=unique_breeds[top_10_pred_indexes]

  #Setup Plot
  top_plot=plt.bar(np.arange(len(top_10_pred_labels)),
                   top_10_pred_values,
                   color="grey")

  plt.xticks(np.arange(len(top_10_pred_labels)),
             labels=top_10_pred_labels,
             rotation="vertical")

  #Change color of the true label
  if np.isin(true_label,top_10_pred_labels):
    top_plot[np.argmax(top_10_pred_labels==true_label)].set_color("green")
  else:
    pass

plot_pred_conf(prediction_probabilities=predictions,  # Corrected spelling
               labels=val_labels,
               n=33)

"""Now 've got some functions to help us visualize our predictions and evaluate oour model, let's check out a few."""

#Let's check out a few predictions and their values

i_multiplier=28
num_rows=3
num_cols=2
num_images=num_rows*num_cols
plt.figure(figsize=(10*num_cols, 5*num_rows))

for i in range(num_images):
  plt.subplot(num_rows,2*num_cols, 2*i+1)
  plot_pred(prediction_probabilities=predictions,
            labels=val_labels,
            images=val_images,
             n=i+i_multiplier)
  plt.subplot(num_rows,2*num_cols, 2*i+2)
  plot_pred_conf(prediction_probabilities=predictions,
                 labels=val_labels,
                 n=i+i_multiplier)
plt.tight_layout(h_pad=1)
plt.show()

"""**Challenge** : How would you create a confusion matrix with our models predictions and truth labels?

# Saving and Reloading a Trained Model
"""

#Create a Function to save a model

def save_model(model,suffix=None):
  '''
  Saves a given model in a models directory and appends a suffix (string).
  '''


  #Create a model directory pathname with current time

  modeldir=os.path.join("/content/model",
                        datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

  model_path=modeldir + "-" + suffix + ".h5" #save format of a model

  print(f"Saving model to : {model_path}...")
  model.save(model_path)

  return model_path

#Create a function to load a trained model
def load_model(model_path):
  '''
  Loads a saved model from a specified path.

  '''

  print(f"Loading Saved model from {model_path}")

  model=tf.keras.models.load_model(model_path,
                                   custom_objects={"KerasLayer":hub.KerasLayer})
  return model

"""Now we've got functions to save and load a trained model, let's make sure they work."""

#Save our model to work on 1000 images.
save_model(model=model,suffix="1000-images-mobilevNet02-Adam")

# Load a Trained Model
loaded_100_image_model=load_model("/content/model/20241229-213511-1000-images-mobilevNet02-Adam.h5")

#Evaluate a pre saved model
model.evaluate(val_data)

#Evaluate the loaded model
loaded_100_image_model.evaluate(val_data)

"""# Training a big dog model 🐶 (on the full dataset)"""

len(X),len(y)

# Create a full data batch with the full data set
full_data=create_data_batches(X,y)

full_data

#Create a model on full data
full_model=create_model()

#Create full model callbacks
full_model_tensorboard=create_tensorboard_callback()

#No validation set when training on all of the data, so we can't monitor accuracy.
full_model_early_stopping=tf.keras.callbacks.EarlyStopping(monitor="accuracy",
                                                            patience=3)

"""**Note** : Running the cell below will take a little while (may be upto 30 minutes on first epoch) because the GPU we'r using in the run time has to load all of the images in the memory."""

#Fit the full model on the full data
full_model.fit(x=full_data,
               epochs=NUM_EPOCHS,
               callbacks=[full_model_tensorboard, full_model_early_stopping])

save_model(full_model,suffix=("full-image-mobilenetv2-Adam"))

#Load the full model
full_model=load_model("/content/model/20241229-214217-full-image-mobilenetv2-Adam.h5")

"""#Make Predictions on the test dataset

Since our model has been trained on images in the form of Tensor Batches , to make predictions on the test data, we'll have to get it in the same format.

Luckily we created 'create_data_batches()' earlier which can take a list of filenames as input and convert them into Tensor Batches.

To make predictions on the test data, we'll:
* Get the test image filenames.
* Convert the filename into test data batches using '' create_data_batches()'' and setting the 'test_data' parameter to 'True'(since the test data parameter does not have labels)
* Make a prediction array by passing the test batches to the 'predict()' method on our model.
"""

# Load test image filenames
test_path="/content/test"
test_filenames=[test_path + fname for fname in os.listdir(test_path)]
test_filenames[:10]

len(test_filenames)

#Create test data batches
test_data=create_data_batches(test_filenames,test_data=True)

test_data

"""** Note **: Since there are 10,000+ test images, making predictions could take a while, even on a GPU. So beware running the cell below may take up to an hour."""

import os

test_path = "/content/test/"  # Include trailing slash
test_filenames = [test_path + fname for fname in os.listdir(test_path) if fname.endswith(".jpg")]
print(test_filenames[:10])  # Verify the constructed paths

import tensorflow as tf
import os

# Define your test directory
test_path = "/content/test/"  # Make sure this directory contains images

# Ensure the directory contains the right structure
print(f"Files in {test_path}:")
print(os.listdir(test_path))

# Create a dataset from your directory
test_data = tf.keras.preprocessing.image_dataset_from_directory(
    test_path,
    label_mode=None,  # We don't need labels for prediction
    image_size=(224, 224),  # Make sure this matches the model's input size
    batch_size=32,  # Adjust batch size as needed
    shuffle=False  # Don't shuffle for prediction
)

# Print the shape of the dataset to verify it's loading correctly
for images in test_data.take(1):
    print(images.shape)  # It should show something like (batch_size, 224, 224, 3)

# Make predictions
predictions = loaded_100_image_model.predict(test_data, verbose=1)

# Output the predictions
print(predictions)

#Make Predictions on the test data batch using loaded full model
test_predictions= loaded_100_image_model.predict(test_data,verbose=1)

#Save Predictions (Numpy Array) to csv file (for later access)
np.savetxt("/content/predictions.csv",test_predictions,delimiter=",")

# Load Predictions (Numpy Array) to a csv file
test_predictions=np.loadtxt("/content/predictions.csv",delimiter=",")

test_predictions[:10]

"""To make predictions on custom images, we will


*   Get the filepaths of our images.

*   Turn the filepath into data batches using create_data_batches(). And since our custom image don't have labels we set the test_data Parameters to True.

*   Pass the Custom image data batch to our model predict() method.

*   Convert the prediction output probabilities to prediction labels.

*   Compare the predicted labels to our custom images.








"""

#Get custom image filepaths
custom_path = "/content/my-dog-photos/"  # Ensure there's a trailing slash
custom_image_paths = [os.path.join(custom_path, fname) for fname in os.listdir(custom_path) if fname.endswith('.jpg')]

custom_image_paths

#Turn custom images into a batch datasets
custom_data=create_data_batches(custom_image_paths,test_data=True)
custom_data

#Make Predictions on Custom Data
custom_preds=loaded_100_image_model.predict(custom_data)

custom_preds

#Get Custom Image prediction labels
custom_pred_labels=[get_pred_label(custom_preds[i]) for i in range(len(custom_preds))]
custom_pred_labels

#Get custom images
custom_images=[]

#Loop through unbatched data
for image in custom_data.unbatch().as_numpy_iterator():
  custom_images.append(image)

#Check Custom Image Predictions
plt.figure(figsize=(10,10))

for i,image in enumerate(custom_images):
  plt.subplot(1,3,i+1)
  plt.xticks([])
  plt.yticks([])
  plt.title(f"Prediction: {custom_pred_labels[i]}")
  plt.imshow(image)