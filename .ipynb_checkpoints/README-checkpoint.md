# Kala Final Project 3 - Predicting wildfire from satellite images

## Built a CNN model and used pre-trained ML models to predict wildfire.
I used a combination of a CNN model and pre-trained ML models to train, test and predict wildfires.

### Data Source - https://www.kaggle.com/datasets/abdelghaniaaba/wildfire-prediction-dataset
### Original Data Source - https://open.canada.ca/data/en/dataset/9d8f219c-4df0-4481-926f-8a2a532ca003
### Creative Commons 4.0 Attribution (CC-BY) license – Quebec - https://www.donneesquebec.ca/licence/

### Step 1: Preprocessing of data
* Resized images to 32X32.
* After converting to a numpy arry, divided all numeric values by 255 for scaling.
* Dataset is very balanced and was already broken out into train, test and validation sets.

### Step 2: CNN Model
* I created a Convolutional Neural Network model as follows:
* 4 sets of 2 convolutional layers (Conv2D) followed by 1 max pooling layer (MaxPooling2D) each
* Then 4 Dense layers.
* Ran this model with a batch size of 32 and epochs between 5 and 10.

### Step 3: Other ML models
#### I also tried the following models:
1. Logistic Regression
1. Random Forest
1. Support Vector Classifier
1. Decision Tree
1. Gradient Boosting
1. ADA Boosting
1. Extra Trees

### Step 4: NYTimes Wildfire Data
* I used the NYTimes API and fetched 10 articles on wildfires.
* I collected the snippets and lead paragraphs of these articles and used spaCy to determine the most common adjectives in them.
* THe most common adjectives found were: dry, strong, rugged, vast, old, dangerous, active and hard.

### Step 5: GenerativeAI
* I used the gemini-1.5-flash model and asked questions related to wildfires.
* I also used ConversationalMemory and asked questions based on initial context.

### Conclusions:
* GenAI is pretty cool!
* CNN model had the highest test and wildfire prediction score - over 96%.
* Random Forest, Gradient Booster and Extra Trees model also performed well, with over 90% scores.

### Sources of code
* Most of my code is based on code provided in exercises across multiple weeks during the AI Bootcamp.