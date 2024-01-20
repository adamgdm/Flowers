# Flowers 
Learning through projects: Flower Recognition Model

### Author: Adam GOUJDAMI 
### Date: 19th January 2024
### Language: Python
### Comments: I know nothing about flowers :D

#### Objective : Objective: Build a model that can classify different types of flowers based on images.

##### The first step to making a machine learning model is Data Collection.

## 1- Data Collection

To start, we need to find a dataset that contains images of flowers. We will use the [Oxford 102 dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) which contains 8189 images of flowers belonging to 102 different categories. The images have large scale, pose and light variations. In addition, there are categories that have large variations within the category and several very similar categories. The dataset is visualized using isomap with shape and colour features. (According to the README file of the dataset)

But before we start, We need to know what datasets are.

---

### What is a dataset?

A dataset, in the context of data science, is a collection of data, often presented in a structured format, that is widely used for analysis, research, training and testing machine learning models. It can contain a wide variety of data types, such as text, numbers, images, audios or videos. 

---

Now that we know what a dataset means, the question that might arise is what is it really? Is it an excel file with a bunch of numerical values? Is it an SQL database with a bunch of tables? Is it a file with some new extension we've never heard of? 

Well the answer is that it can be any of these things. A dataset is simply just a collection of data. Be it a CSV file, a database, a JSON file or any other file type. It can even be a collection of files. The only requirement is that it contains data.

##### Alright, now that we know we know what a dataset and have one, The next step is Data Preprocessing.

## 2- Data Preprocessing

---

### What is Data Preprocessing?

Data preprocessing is the process of converting data from the inital *raw* form into clean and concise data. It involves numerous steps such as cleaning, encoding, imputing, transforming and scaling. This Step is very important and crucial to the success of the model as it can make or break it. 

---

In our case, our dataset is clean and ready to be used. 

The only thing we need to do is resizing the images to a standard size. We will use the `cv2` library to do this. We'll create a new file and call it `resize.py` and add the following code to it:

```python
# Importing the libraries
import cv2

# define a function that will resize the images
def resize(img_path, target_size):
	# Set the target size if it is not set
	if target_size is None:
		target_size = (256, 256)
	
	# Read the image
	img = cv2.imread(img_path)
	
	# Resize the image
	img = cv2.resize(img, target_size)

	# Return the resized image
	return img
```

We will use this function to resize the images when we load them in the model.

## 3- Data Exploration

---

### What is Data Exploration?

Data exploration is the process of visualizing some of the features of the dataset to get a better understanding of it. It is a very important step in the process of building a machine learning model as it helps us understand the data better and make better decisions about how to process it.

---

In our case, I kindly invite you to explore the dataset and visualize some of its features. It is indeed a very colorful dataset with beautiful flowers. and I'm sure you'll enjoy it.

Now that we have explored the dataset, we can move on to the next step which is Model Building.

## 4- Model Building

##### Disclaimer: This is going to be a very very long section. So grab a cup of coffee and get ready to read a lot of text.

---

### What is Model Building?

Model building is the process of creating and defining the architecture of the machine learning model. It involves selecting a specific type of model, defining its architecture, configuring its parameters and compiling it so that it can be trained and used for making predictions.

This step is very important as it is the core of the machine learning process. It is where the model is defined and created. It is also the step where most of the work is done. 

There are a lot of things to be done in this particular phase. The important and crucial ones are:

#### 1- Selecting a model

#### 2- Defining the model architecture

#### 3- Configuring the model parameters and hyperparameters

#### 4- Compiling the model

#### 5- Summarizing the model for better understanding

---

Now if this seems like gibberish to you, don't worry. It does to me too. But we'll go through each of these steps one by one and explain them in too much detail you might get bored. (i hope not :D)

Okay, so let's go back to our definition of Model Building. We said that it is the process of creating and defining the architecture of the machine learning model. So what does that mean?

What is a machine learning model? What is its architecture? What does it mean to define the architecture of a machine learning model? Well, let's answer these questions one by one.

#### What is a machine learning model?

A machine learning model is a computational algorithm that is trained on data to make predictions or decisions without being explicitly programmed to do so. It is a mathematical representation of the data that is used to make predictions or decisions.

Simply put, a machine learning model is a smart computer program that learns from examples to predict or decide things without needing direct instructions from a human. It's like teaching the computer to make a smart decision by showing it tons of examples.

Let's Jump to the next question and deep dive into the architeture of a machine learning model.

#### What is the architecture of a machine learning model?

The architecture of a machine learning model refers to the structure or the design of the particular model. It is the way the model is designed and built, including the arrangement and configuration of its components. Components such as layers, neurons, activation functions, loss functions, optimizers, etc. (Gibberish again, I know. But don't worry, we'll explain all of these in detail in the next sections)

To put it in simple terms, the architecture of a machine learning model can be thought of as the blueprint of the model.

There are a lot of different types of machine learning models. each with its own architecture. The most common ones are: 

##### - Artificial Neural Networks (ANNs)
##### - Convolutional Neural Networks (CNNs)
##### - Recurrent Neural Networks (RNNs)
##### - Long Short Term Memory Networks (LSTMs)
##### - Gated Recurrent Unit Networks (GRUs)
##### - Autoencoders
##### - Transformers
##### - Generative Adversarial Networks (GANs)
##### - Support Vector Machines (SVMs)
##### - Decision Trees
##### - Random Forests
##### - K-Nearest Neighbors (KNNs)

And many many more. In this project we will be using a Convolutional Neural Network (CNN) as it is the most commonly used model for image classification tasks. 

---

We will try to break down all the other important Models in other projects but for now, let's focus on CNNs. Get ready for some more gibberish.

#### What is a Convolutional Neural Network (CNN)?

To start, let's understand the name of the model. 

##### Convolutional : Thinks in small parts. Like looking at small pieces of puzzle instead of the whole puzzle at once.

##### Neural : It is a neural network. It is made up of neurons, inspired by the human brain. Uses layers of neurons to learn and understand. Each layers gets better at understanding the data.

##### Network : It is a network of layers. Each layer is connected to the next layer. The output of one layer is the input of the next layer.

So, a Convolutional Neural Network is a class of artificial neural networks that is most commonly used for image classification tasks. It is designed to think in small parts and learn from examples.

We could think of it as detectives (neural network) looking at small pieces (convolutional) of a case (data, images in this case). 

Now that we know what a CNN is, let's understand how it works.

#### How does a Convolutional Neural Network (CNN) work?

To understand how a CNN works, we need to understand how a neural network works. So let's start with that.

The human brain is made up of billions of neurons. These neurons are connected to each other in a network. This network of neurons is what allows us to think, learn and understand. this last inspired the creation of artificial neural networks.

Similarly to the human brain, an artificial neural network is made up of neurons that are connected to each other in a network. This network of neurons is what allows the model to think, learn and understand.

- The Neurons : The fundamental building blocks of a neural network. Each neuron is a mathematical function that takes an input and produces an output. The output of one neuron is the input of the next neuron. and so on...

These neurons are organized in layers. 

- The Layers : The building blocks of a neural network. Each layer is a collection of neurons that are connected to each other. 

	There are 3 types of layers:

	- Input Layer : The first layer of the neural network. It is the layer that receives the input data. It stands as the sole layer directly connected to the model's input.

	- Hidden Layers : The layers that are between the input layer and the output layer. They are called hidden layers because the user does not have direct access to them. They are the layers that do all the work.

	- Output Layer : The last layer of the neural network, responsible for generating the model's output. It stands as the exclusive layer directly connected to the model's output.

Each layer is made up of neurons that are connected to each other. 

- The Connections : They are the links between neurons, allowing them to communicate with each other. They are represented by weights (Mathermatical values) that are assigned to each connection. These weights are what allow the model to learn and understand (More on weights coming up now).

- The Weights : Mathematical values assigned to each connection between neurons. They represent the strength of the connection between neurons. They are what allow the model to learn and understand. (I know, It's not very clear, But we'll understand it better in an example coming up soon)

With Weights, comes Bias. 

- The Bias : Additional parameter (Mathematical value) added to each neuron to enhance and improve the model's performance. It is used to adjust and fine-tune the output of each neuron. (Again, not very clear, But Don't worry.)

All of this is then combined and passed through an activation function.

- The Activation Function : A mathematical function that is applied to the output of each neuron. It is used to introduce non-linearity into the model. (We'll get to non-linearity later on)

---

Let's take a look at an example to understand all of this better:

```
Input Layer : Single Input (x1) 
	|
	Connection : Weight (W1) and Bias (B1)
	|
	└--> Hidden Layer : 1 Neuron, Takes Input (X1), Applies
		Weight (W1) and Bias (B1) to it and Produces output (Y1)
		* Y1 = W1 * X1 + B1

		The Sum Y1 is then passed through an Activation Function (F1) to produce the final output (Z1)
		* Z1 = F1(Y1)
		Z1 is the output of the Hidden Layer
		|
		Connection : Weight (W2) and Bias (B2)
		|
		└--> Output Layer : Takes Input (Z1), Applies Weight (W2)
			and bias (B2) to it and Produces Output (Y2)
			* Y2 = W2 * Z1 + B2

			The Sum Y2 is then passed through an Activation Function (F2) to produce the final output (Z2)
			* Z2 = F2(Y2)
			Z2 is the final output of the model (The Prediction)

	To visualize this neural network: 

	  Neuron 1					    Neuron 2                               Neuron 3
	/‾‾‾‾‾‾‾‾‾‾\		       /‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\			   /‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\
	| Input X1 |---(W1, B1)--->| f(X1*W1 + B1) |---(W2, B2)--->| Prediction Z2 = f(Z1*W2 + B2) |
	\__________/			   \_______________/			   \_______________________________/

	Where:
		- X1 is the input
		- W1 is the weight of the connection 1
		- B1 is the bias of the connection 1
		- f is the activation function
		- Z1 is the output of the hidden layer
		- W2 is the weight of the connection 2
		- B2 is the bias of the connection 2
		- Z2 is the final output (The Prediction)

	This is a very very simple neural network. It has only 1 input, 1 hidden layer and 1 output. We should keep in mind that typically, a neural network has multiple inputs, multiple hidden layers and multiple outputs, Ranging from a few to hundreds of thousands.

```
---

	![Neural Network](https://miro.medium.com/max/1400/1*ZB6H4HuF58VcMOWbdpcRxQ.jpeg)

Let's not forget about the activation function and non-linearity.

Like we said before, the activation function is a mathematical function that is applied to the output of each neuron. It is used to introduce non-linearity into the model.

##### What is non-linearity?

Simply put, non-linearity is the property of a mathematical function that is not a straight line, therefore not linear.

To simplify it even more, The relationship between the input X and the out Y is not linear. 

A linear relationship between X and Y is expressed as : Y = aX + b where a (the slope) and b (the intercept) are constants and the slightest change in X will result in a proportional change in Y.

Non linear relationships exhibit more complex behaviors. They can take various forms such as exponential, logarithmic, quadratic, etc. and the slightest change in X could result in a non proportional change or an exponential one in Y.

![Linear vs Non-Linear](https://miro.medium.com/max/1400/1*QJZ6W-Pck_W7RlIDwUIN9Q.png)

Alright, now that we understand how a neural network works, let's go back to CNNs.

The CNN's architecture is very similar to that of a neural network:

![CNN Architecture](https://miro.medium.com/max/1400/1*vkQ0hXDaQv57sALXAJquxA.jpeg)