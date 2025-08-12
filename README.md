# Machine Learning Regularization: L1 vs L2 on High-Dimensional Data

This project demonstrates the application and comparison of **L1 (Lasso)** and **L2 (Ridge)** regularization techniques on a high-dimensional dataset (p >> n).  
The workflow focuses on evaluating model performance, tuning hyperparameters, and visualizing results to choose a generalized model.

> **Note:** Due to privacy concerns, the original schizophrenia dataset is not shared in this repository.  
Instead, a **synthetic dataset** with the same structure is provided for demonstration purposes.

---

##  Project Structure

Project: Repo/
├── data/
│ ├── synthetic_features.csv # Synthetic Schizophrenia gene expression features of  
│ ├── synthetic_labels.csv # Synthetic binary labels (0/1)
├── src/
│ ├── Script.py # Main script
  ├── Code.ipynb # Jupyter Notebook
├── README.md

##  Methods
1. Load dataset (synthetic for demo).
2. Split into training and testing sets.
3. Train logistic regression models with:
   - **L1 regularization** (`penalty='l1'`)
   - **L2 regularization** (`penalty='l2'`)
4. Perform hyperparameter tuning on `lambda` (regularization strength).
5. Compare accuracy and generalization.
6. Visualize coefficient paths and performance metrics.


##  Results
- Plotted model accuracy against different lambda values.
- Identified optimal lambda for a well-generalized model.
- Demonstrated difference between sparse (L1) and dense (L2) coefficient patterns.

## Author:
- Hira
- MS. Life Science Informatics
- University of Bonn
