# Project3_Wildfire-Wildfire Prediction from Satellite Images

## Project Overview
This project aims to predict wildfires using a combination of a Convolutional Neural Network (CNN) model and various machine learning algorithms. Additionally, Natural Language Processing (NLP) and Generative AI models are utilized to analyze wildfire-related content and simulate contextual question-answering. By integrating diverse methods, this project provides a holistic approach to understanding and addressing wildfire predictions and prevention.

## Data Sources
- **Wildfire Prediction Dataset**: [Kaggle Dataset](https://www.kaggle.com/datasets/abdelghaniaaba/wildfire-prediction-dataset)
  - Original Source: [Canadian Open Data Portal](https://open.canada.ca/data/en/dataset/9d8f219c-4df0-4481-926f-8a2a532ca003)
  - License: Creative Commons Attribution (CC-BY) 4.0 Quebec - [Details Here](https://www.donneesquebec.ca/licence/)

## Project Pipeline
1. **Data Preprocessing**
   - Resized images to 32x32 pixels.
   - Converted images to NumPy arrays and scaled numeric values by dividing by 255 for normalization.
   - Dataset is balanced and pre-divided into training, testing, and validation sets.

2. **Model Training**
   - **Convolutional Neural Network (CNN):**
     - **Architecture**:
       - 4 convolutional layers (Conv2D) paired with max-pooling layers (MaxPooling2D).
       - 4 dense layers for classification.
     - **Parameters**:
       - Batch size: 32
       - Epochs: 5-10
     - **Performance**: Achieved a test accuracy of over 96%.

   - **Additional Machine Learning Models**:
     - Implemented and evaluated the following models using a standardized pipeline:
       - Logistic Regression
       - Random Forest
       - Support Vector Classifier (SVC)
       - Decision Tree
       - Gradient Boosting
       - AdaBoost
       - Extra Trees
     - **Performance**: Random Forest, Gradient Boosting, and Extra Trees achieved test scores exceeding 90%.

3. **NLP Analysis**
   - Utilized the New York Times API to retrieve 10 wildfire-related articles.
   - Extracted snippets and lead paragraphs, analyzed using spaCy.
   - Identified the most common adjectives describing wildfires: "dry," "strong," "rugged," "vast," "old," "dangerous," "active," and "hard."

4. **Generative AI Integration**
   - Leveraged the Gemini-1.5-flash model for contextual Q&A related to wildfires.
   - Employed Conversational Memory for dynamic responses based on initial queries.
   - Results highlight the potential of Generative AI in enhancing wildfire awareness and preparedness.

## Code Structure
### Key Notebooks
1. **`fires.ipynb`**: Implements the CNN model for wildfire prediction.
2. **`firms_data_ingest.ipynb`**: Prepares and preprocesses the dataset for model training.
3. **`is_it_wildfire.ipynb`**: Validates predictions using additional ML models.
4. **`OptimizeModel.ipynb`**: Fine-tunes hyperparameters and evaluates performance metrics across models.

### Pipeline Utilities
The `pipeline_utilities.py` script provides modular functions for model training and evaluation:
- **Scaling**: StandardScaler for normalization.
- **Model Generators**:
  - Logistic Regression, Random Forest, SVC, Decision Tree, Gradient Boosting, AdaBoost, Extra Trees.
- **Metrics**:
  - Accuracy Score, Balanced Accuracy, ROC-AUC, and Classification Reports.
- **Reusable Components**: Functions enable seamless integration across notebooks.

## Conclusions
- The CNN model demonstrated the highest accuracy (96%+) in wildfire prediction.
- Ensemble models (Random Forest, Gradient Boosting, Extra Trees) performed robustly with over 90% accuracy.
- NLP and Generative AI tools provided valuable insights into wildfire descriptions and potential preventive measures.
- Modular code design ensures scalability for future enhancements.

## Sources and References
- Core CNN implementation adapted from [Kaggle Code: Wildfire CNN Accuracy 95%](https://www.kaggle.com/code/yassinosama911/wild-fire-cnn-accuracy-95).
- Machine learning pipeline inspired by exercises from the AI Bootcamp.

---

This project reflects a collaborative effort to explore innovative approaches to wildfire prediction and prevention using cutting-edge technologies. The integration of visual data analysis, linguistic insights, and AI-driven solutions exemplifies the potential for data-driven disaster management.
