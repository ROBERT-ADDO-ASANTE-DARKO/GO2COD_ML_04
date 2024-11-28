# Iris Flower Classification

This project focuses on classifying iris flowers into one of three species: *Setosa*, *Versicolor*, and *Virginica*, using machine learning algorithms. The dataset used is the classic Iris dataset, which is widely used for beginner machine learning tasks. Hyperparameter tuning techniques, such as GridSearchCV, were employed to optimize model performance.

## 📂 Project Structure

- `Iris_Flower_Classification.ipynb`: Jupyter notebook containing the complete workflow, including data preprocessing, model training, evaluation, and hyperparameter tuning.
- `data/`: Directory for storing the Iris dataset (if not loaded directly from a library).

## 🚀 Features

1. **Data Preprocessing**:
   - Analysis of the Iris dataset.
   - Scaling features for better model performance.
   - Splitting the dataset into training and test sets.

2. **Exploratory Data Analysis**:
   - Visualization of feature distributions.
   - Pairwise relationships between features.

3. **Machine Learning Models**:
   - Multiple algorithms implemented:
     - Logistic Regression
     - Support Vector Machines (SVM)
     - K-Nearest Neighbors (KNN)
     - Decision Trees
     - Random Forests
   - **Hyperparameter Tuning**:
     - Used GridSearchCV to find the optimal parameters for each model.

4. **Evaluation Metrics**:
   - Accuracy
   - Confusion Matrix
   - Precision, Recall, and F1-Score

5. **Comparison of Models**:
   - Benchmarked different algorithms to identify the best-performing model for this classification task.

## 📊 Dataset
The Iris dataset consists of 150 samples, with the following features:
- Sepal length
- Sepal width
- Petal length
- Petal width

Target variable: Species (*Setosa*, *Versicolor*, *Virginica*)

The dataset is available in popular libraries like Scikit-learn or can be downloaded from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/iris).

## 🛠️ Tools and Libraries
The project utilizes the following Python libraries:
- **Pandas**: Data manipulation and analysis.
- **Numpy**: Numerical computations.
- **Matplotlib** & **Seaborn**: Data visualization.
- **Scikit-learn**: Machine learning models, metrics, and hyperparameter tuning.

## ⚙️ How to Run
1. Clone the repository.
   ```bash
   git clone https://github.com/ROBERT-ADDO-ASANTE-DARKO/GO2COD_ML_04.git
   ```
2. Open the `Iris_Flower_Classification.ipynb` notebook in Jupyter or any Python IDE supporting notebooks.
3. Install the required libraries directly within the notebook:
   ```python
   !pip install pandas numpy scikit-learn matplotlib seaborn
   ```
4. Run the cells sequentially to execute the workflow.

## 📈 Model Performance
| Model               | Best Hyperparameters         | Accuracy |
|---------------------|------------------------------|----------|
| Logistic Regression | Grid-searched parameters     | 97.78%      |
| SVM                 | Grid-searched parameters     | 97.78%      |
| KNN                 | Grid-searched parameters     | 97.78%      |
| Decision Tree       | Grid-searched parameters     | 95.56%      |
| Random Forest       | Grid-searched parameters     | 97.78%      |

## 📌 Highlights
- **Hyperparameter Tuning**: Enhanced model performance with GridSearchCV.
- **Visualization**: Explored the relationships between features and species with pair plots and histograms.
- **Model Benchmarking**: Compared multiple algorithms to select the most suitable one.

## 🔗 Acknowledgments
- The Iris dataset, a classic in machine learning, is provided by UCI Machine Learning Repository.
- Thanks to Scikit-learn for its comprehensive machine learning tools.
