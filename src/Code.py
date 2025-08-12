{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "780d9c88",
   "metadata": {},
   "source": [
    "### Machine Learning Model Regularization (L1 and L2)\n",
    "In this project we will compare the accuracy of different regularization a high-dimensional (p>>n) dataset. "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "db865260",
   "metadata": {},
   "source": [
    "(A): Load the Schizophrenia dataset (schizophrenia_data.csv) and extract the class labels from the schizophrenia_labels.csv table (2 classes, “control” and “schizophrenia”). Randomly split the data into 70% training and 30% test. (Hint: Use the train_test_split function from scikit-learn to define the test_size and set random_state=1 for better reproducibility.)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "aa8b98f8",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "     Gene_1    Gene_2    Gene_3    Gene_4    Gene_5    Gene_6    Gene_7  \\\n",
      "0  3.107048  2.069152  7.111179  7.679008  9.401203  7.222667  3.803919   \n",
      "1  5.407296  9.584915  9.898940  5.885055  5.293680  8.671617  3.860081   \n",
      "2  0.622331  5.008331  1.389514  2.120394  4.797003  2.614428  4.230783   \n",
      "3  7.936438  9.546866  7.543264  8.365743  6.282611  9.666334  1.816157   \n",
      "4  1.617207  0.140010  6.102569  4.311913  6.917246  1.857639  3.517117   \n",
      "\n",
      "     Gene_8    Gene_9   Gene_10  ...   Gene_91   Gene_92   Gene_93   Gene_94  \\\n",
      "0  8.117723  0.698969  9.904075  ...  4.273160  2.020639  8.307424  5.453386   \n",
      "1  8.345644  4.412550  1.855874  ...  2.635558  0.171963  6.181632  9.205937   \n",
      "2  3.386910  1.736005  9.463612  ...  4.485451  7.761712  7.460294  5.509614   \n",
      "3  4.191232  6.343884  1.772025  ...  7.531534  6.002260  7.604890  2.994918   \n",
      "4  8.089752  9.501443  1.247513  ...  5.721809  1.062997  1.827951  0.713860   \n",
      "\n",
      "    Gene_95   Gene_96   Gene_97   Gene_98   Gene_99  Gene_100  \n",
      "0  8.903415  5.081454  0.989369  2.027119  7.036629  2.927711  \n",
      "1  2.584836  7.720987  2.464948  8.641086  5.875631  0.201306  \n",
      "2  6.052722  9.968873  0.027643  2.852324  0.726559  1.779592  \n",
      "3  1.567732  5.592238  2.011023  2.144252  9.797812  3.340708  \n",
      "4  7.977235  4.758137  5.988468  4.754606  6.223947  6.393305  \n",
      "\n",
      "[5 rows x 100 columns]\n",
      "Class counts before splitting:\n",
      "Target\n",
      "0    6\n",
      "1    4\n",
      "Name: count, dtype: int64\n",
      "\n",
      "Class counts after splitting:\n",
      "\n",
      "Training data class counts:\n",
      "Target\n",
      "0    4\n",
      "1    3\n",
      "Name: count, dtype: int64\n",
      "\n",
      "Test data class counts:\n",
      "Target\n",
      "0    2\n",
      "1    1\n",
      "Name: count, dtype: int64\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "\n",
    "# Load the data and label\n",
    "data_df = pd.read_csv(\"../data/synthetic_features_data.csv\")  # synthetic data file \n",
    "labels = pd.read_csv(\"../data/synthetic_labels.csv\")  # synthetic label file\n",
    "data = data_df.select_dtypes(include=['int64', 'float64'])\n",
    "\n",
    "\n",
    "# Print \n",
    "print(data.head())\n",
    "print(\"Class counts before splitting:\")\n",
    "print(labels[\"Target\"].value_counts())\n",
    "\n",
    "# Simple split (70% training data, 30% test data)\n",
    "X_train, X_test, Y_train, Y_test = train_test_split(\n",
    "    data,                      # Features data\n",
    "    labels[\"Target\"],          # Just the 'Target' column (correspond label)\n",
    "    test_size=0.3,             # 30% for testing data\n",
    "    random_state=1             # results reproducible\n",
    ")\n",
    "\n",
    "# print label splitting as an example\n",
    "print (\"\\nClass counts after splitting:\")\n",
    "print(\"\\nTraining data class counts:\")\n",
    "print(Y_train.value_counts())\n",
    "print(\"\\nTest data class counts:\")\n",
    "print(Y_test.value_counts())\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "47e87e1f",
   "metadata": {},
   "source": [
    "(B): Fit a logistic regression (no penalization)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "c503a11a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Unpenalized Logistic Regression Model Done on Train data\n"
     ]
    }
   ],
   "source": [
    "from sklearn.linear_model import LogisticRegression\n",
    "\n",
    "# initialize the model \n",
    "model = LogisticRegression (penalty=None, max_iter=1000, random_state=1) # Set no regularization, more iteration so ensure convergence, reproducible result\n",
    "\n",
    "# Fit model on train data\n",
    "model_no_penalty = model.fit(X_train, Y_train)\n",
    "\n",
    "# print\n",
    "print (\"Unpenalized Logistic Regression Model Done on Train data\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d6df5837",
   "metadata": {},
   "source": [
    "(C):  Fit multiple l1-penalized logistic regressions (lambdas = 0.001, 0.01, 0.1, 1, 10, 100)\n",
    "Hint: Please read the documentation and choose a solver that would work well with your dataset and penalization method. \n",
    "Hint: Scikit-learn uses a parameter called C, where C = 1/lambda.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "69a3d311",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "C values: [1000.0, 100.0, 10.0, 1.0, 0.1, 0.01]\n",
      "model_L1 has done on train data as per lambda strength\n",
      "model_L1 has done on train data as per lambda strength\n",
      "model_L1 has done on train data as per lambda strength\n",
      "model_L1 has done on train data as per lambda strength\n",
      "model_L1 has done on train data as per lambda strength\n",
      "model_L1 has done on train data as per lambda strength\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "\n",
    "# Lambda values (penalty strengths)\n",
    "lambdas = [0.001, 0.01, 0.1, 1, 10, 100]  \n",
    "\n",
    "# Convert to C (C = 1/lambda)\n",
    "Cs = [1/l for l in lambdas]  \n",
    "print (\"C values:\", Cs)\n",
    "\n",
    "# Store all trained models\n",
    "models_L1 = {} \n",
    "\n",
    "for C in Cs:\n",
    "    # Initialize model with L1 penalty\n",
    "    model = LogisticRegression(\n",
    "        penalty=\"l1\",       # Lasso regression (L1 regularization)\n",
    "        C=C,                # Inverse of lambda\n",
    "        solver=\"liblinear\", # liblinear use here because this suits for L1 (in our dataset, more feature less sample)\n",
    "        max_iter=10000,     # Ensure convergence\n",
    "        random_state=1      # Reproducibility\n",
    "    )\n",
    "    \n",
    "    # Fit model on train data \n",
    "    models_L1[f\"λ={1/C:.3f}\"] = model.fit(X_train, Y_train)\n",
    "    print (\"model_L1 has done on train data as per lambda strength\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "23428043",
   "metadata": {},
   "source": [
    "(D):  Fit multiple l2-penalized logistic regressions (lambdas = 0.001, 0.01, 0.1, 1, 10, 100) (2 points)\n",
    "Hint: Please read the documentation and choose a solver that would work well with your dataset and penalization method.\n",
    "Hint: Scikit-learn uses a parameter called C, where C = 1/lambda.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "76d69fbd",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "C values: [1000.0, 100.0, 10.0, 1.0, 0.1, 0.01]\n",
      "Model L2 has done on train data on all lambda strength\n",
      "Model L2 has done on train data on all lambda strength\n",
      "Model L2 has done on train data on all lambda strength\n",
      "Model L2 has done on train data on all lambda strength\n",
      "Model L2 has done on train data on all lambda strength\n",
      "Model L2 has done on train data on all lambda strength\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "\n",
    "# lambda values (strength penalty)\n",
    "lambdas = [0.001, 0.01, 0.1, 1, 10, 100]\n",
    "\n",
    "# Convert to C (C = 1\\lambda)\n",
    "Cs = [1/l for l in lambdas]\n",
    "print (\"C values:\", Cs)\n",
    "\n",
    "# Store all trained models as per lambda\n",
    "models_L2 = {}\n",
    "\n",
    "for C in Cs:\n",
    "    # Initialize model with L2 penalty\n",
    "    model = LogisticRegression(\n",
    "        penalty=\"l2\",          # Ridge Regression (L2 Regularization)\n",
    "        C=C,                   # C = 1/lambda\n",
    "        solver='lbfgs',        # Recommended solver for L2\n",
    "        max_iter=10000,        # Ensure convergence\n",
    "        random_state=1         # Reproducibility\n",
    "    )\n",
    "    \n",
    "    # Fit model on train data\n",
    "    models_L2[f\"λ={1/C:.3f}\"] = model.fit(X_train, Y_train)\n",
    "    print (\"Model L2 has done on train data on all lambda strength\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e8872e5b",
   "metadata": {},
   "source": [
    "(E):  For the models from (b), (c), and (d) measure the performance in terms of accuracy, F1 Score, AUROC and average precision recall score on the training and test set.\n",
    "Hint: For the areas under the curves (AUROC / AUPR), we need to consider predicted class probabilities instead of only binarized class predictions.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "021b28d1",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Unpenalized Logistic Regression:\n",
      "Training: {'Accuracy': 1.0, 'F1': 1.0, 'AUROC': 1.0, 'AUPR': 1.0}\n",
      "Test: {'Accuracy': 0.3333333333333333, 'F1': 0.0, 'AUROC': 0.5, 'AUPR': 0.5}\n",
      "\n",
      "L1-Penalized Logistic Regression:\n",
      "\n",
      "Lambda: λ=0.001\n",
      "Training: {'Accuracy': 1.0, 'F1': 1.0, 'AUROC': 1.0, 'AUPR': 1.0}\n",
      "Test: {'Accuracy': 0.6666666666666666, 'F1': 0.0, 'AUROC': 0.5, 'AUPR': 0.5}\n",
      "\n",
      "Lambda: λ=0.010\n",
      "Training: {'Accuracy': 1.0, 'F1': 1.0, 'AUROC': 1.0, 'AUPR': 1.0}\n",
      "Test: {'Accuracy': 0.3333333333333333, 'F1': 0.0, 'AUROC': 0.0, 'AUPR': 0.3333333333333333}\n",
      "\n",
      "Lambda: λ=0.100\n",
      "Training: {'Accuracy': 1.0, 'F1': 1.0, 'AUROC': 1.0, 'AUPR': 1.0}\n",
      "Test: {'Accuracy': 0.3333333333333333, 'F1': 0.0, 'AUROC': 0.0, 'AUPR': 0.3333333333333333}\n",
      "\n",
      "Lambda: λ=1.000\n",
      "Training: {'Accuracy': 1.0, 'F1': 1.0, 'AUROC': 1.0, 'AUPR': 1.0}\n",
      "Test: {'Accuracy': 0.6666666666666666, 'F1': 0.0, 'AUROC': 0.0, 'AUPR': 0.3333333333333333}\n",
      "\n",
      "Lambda: λ=10.000\n",
      "Training: {'Accuracy': 0.7142857142857143, 'F1': 0.5, 'AUROC': 1.0, 'AUPR': 1.0}\n",
      "Test: {'Accuracy': 0.6666666666666666, 'F1': 0.0, 'AUROC': 0.0, 'AUPR': 0.3333333333333333}\n",
      "\n",
      "Lambda: λ=100.000\n",
      "Training: {'Accuracy': 0.5714285714285714, 'F1': 0.0, 'AUROC': 0.5, 'AUPR': 0.42857142857142855}\n",
      "Test: {'Accuracy': 0.6666666666666666, 'F1': 0.0, 'AUROC': 0.5, 'AUPR': 0.3333333333333333}\n",
      "\n",
      "L2-Penalized Logistic Regression:\n",
      "\n",
      "Lambda: λ=0.001\n",
      "Training: {'Accuracy': 1.0, 'F1': 1.0, 'AUROC': 1.0, 'AUPR': 1.0}\n",
      "Test: {'Accuracy': 0.3333333333333333, 'F1': 0.0, 'AUROC': 0.5, 'AUPR': 0.5}\n",
      "\n",
      "Lambda: λ=0.010\n",
      "Training: {'Accuracy': 1.0, 'F1': 1.0, 'AUROC': 1.0, 'AUPR': 1.0}\n",
      "Test: {'Accuracy': 0.3333333333333333, 'F1': 0.0, 'AUROC': 0.5, 'AUPR': 0.5}\n",
      "\n",
      "Lambda: λ=0.100\n",
      "Training: {'Accuracy': 1.0, 'F1': 1.0, 'AUROC': 1.0, 'AUPR': 1.0}\n",
      "Test: {'Accuracy': 0.3333333333333333, 'F1': 0.0, 'AUROC': 0.5, 'AUPR': 0.5}\n",
      "\n",
      "Lambda: λ=1.000\n",
      "Training: {'Accuracy': 1.0, 'F1': 1.0, 'AUROC': 1.0, 'AUPR': 1.0}\n",
      "Test: {'Accuracy': 0.3333333333333333, 'F1': 0.0, 'AUROC': 0.5, 'AUPR': 0.5}\n",
      "\n",
      "Lambda: λ=10.000\n",
      "Training: {'Accuracy': 1.0, 'F1': 1.0, 'AUROC': 1.0, 'AUPR': 1.0}\n",
      "Test: {'Accuracy': 0.3333333333333333, 'F1': 0.0, 'AUROC': 0.5, 'AUPR': 0.5}\n",
      "\n",
      "Lambda: λ=100.000\n",
      "Training: {'Accuracy': 1.0, 'F1': 1.0, 'AUROC': 1.0, 'AUPR': 1.0}\n",
      "Test: {'Accuracy': 0.6666666666666666, 'F1': 0.0, 'AUROC': 0.5, 'AUPR': 0.5}\n"
     ]
    }
   ],
   "source": [
    "from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score\n",
    "\n",
    "def evaluate_model(model, X, y_true):\n",
    "    y_probs = model.predict_proba(X)[:, 1]  # Positive class probabilities\n",
    "    y_pred = model.predict(X)\n",
    "    return {\n",
    "        'Accuracy': accuracy_score(y_true, y_pred),\n",
    "        'F1': f1_score(y_true, y_pred, pos_label=1),\n",
    "        'AUROC': roc_auc_score(y_true, y_probs),\n",
    "        'AUPR': average_precision_score(y_true, y_probs, pos_label=1)\n",
    "    }\n",
    "\n",
    "# First model (no penalty)\n",
    "print(\"\\nUnpenalized Logistic Regression:\")\n",
    "train_metrics = evaluate_model(model_no_penalty, X_train, Y_train)\n",
    "test_metrics = evaluate_model(model_no_penalty, X_test, Y_test)\n",
    "print(\"Training:\", train_metrics)\n",
    "print(\"Test:\", test_metrics)\n",
    "\n",
    "\n",
    "# L1 penalized model on different lambda\n",
    "print(\"\\nL1-Penalized Logistic Regression:\")\n",
    "for lambda_value, model in models_L1.items():  \n",
    "    print(f\"\\nLambda: {lambda_value}\")\n",
    "    train_metrics = evaluate_model(model, X_train, Y_train)\n",
    "    test_metrics = evaluate_model(model, X_test, Y_test)\n",
    "    print(\"Training:\", train_metrics)\n",
    "    print(\"Test:\", test_metrics)\n",
    "   \n",
    "\n",
    "# L2 penalized model on different lambda\n",
    "print(\"\\nL2-Penalized Logistic Regression:\")\n",
    "for lambda_value, model in models_L2.items(): \n",
    "    print(f\"\\nLambda: {lambda_value}\")\n",
    "    train_metrics = evaluate_model(model, X_train, Y_train)\n",
    "    test_metrics = evaluate_model(model, X_test, Y_test)\n",
    "    print(\"Training:\", train_metrics)\n",
    "    print(\"Test:\", test_metrics)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "440da95f",
   "metadata": {},
   "source": [
    "Report the accuracy with one barplot for each approach (1 barplot for unpenalized, l1, l2), with the regularization constant on the x-axis and the accuracy on the y-axis, train and test set colored differently.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "99403391",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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nDkDDw8PuU1HRwceHh6IjY0t0jJevHiBrKwsVK9evbTKJCL6oFTS5sqfPHmCnJwcWFhYCO0WFha4evVqkZbx3XffwcrKSgiXN2VkZCAjI0OeTklJKX7BREQfAK3vSvo75s2bh6ioKOzcuRP6+vr59gkJCYFKpZJv1tbWZVwlEdH7RavBYGpqCl1dXTx69Ehof/ToESwtLd8578KFCzFv3jz89ttvaNasWYH9pkyZguTkZPl27969EqmdiKii0mow6OnpwcXFRRg4zh1IbtOmTYHzLViwALNmzcL+/fvRsmXLd65DqVTCxMREuBERUcG0OsYAAIGBgfDz80PLli3RqlUrhIWFIT09HQEBAQAAX19f1KpVCyEhIQCA+fPnY8aMGdi0aRNsbW2RmJgIADAyMoKRkZHWHgcRUUWh9WAYMGAAkpKSMGPGDCQmJsLZ2Rn79++XB6QTEhKgo/O/DZtly5YhMzMT/fr1E5YTFBSEmTNnlmXpREQVktaDAQDGjh2LsWPH5nvfkSNHhOn4+PjSL4iI6AP2Xh+VREREJY/BQEREAgYDEREJGAxERCRgMBARkYDBQEREAgYDEREJGAxERCRgMBARkYDBQEREAgYDEREJGAxERCRgMBARkYDBQEREAgYDEREJGAxERCRgMBARkYDBQEREAgYDEREJGAxERCRgMBARkYDBQEREAgYDEREJGAxERCRgMBARkYDBQEREAgYDEREJGAxERCRgMBARkYDBQEREAgYDEREJGAxERCRgMBARkYDBQEREAgYDEREJGAxERCRgMBARkYDBQEREAgYDEREJGAxERCRgMBARkYDBQEREAgYDEREJGAxERCRgMBARkYDBQEREAgYDEREJGAxERCRgMBARkYDBQEREAgYDEREJGAxERCRgMBARkYDBQEREAgYDEREJGAxERCRgMBARkYDBQEREAgYDEREJGAxERCRgMBARkYDBQEREAgYDEREJGAxERCRgMBARkYDBQEREAgYDEREJGAxERCRgMBARkYDBQEREgnIRDEuWLIGtrS309fXRunVrnDp16p39t23bBkdHR+jr66Np06bYu3dvGVVKRFTxaT0YtmzZgsDAQAQFBeHs2bNwcnKCp6cnHj9+nG//EydOYNCgQRg2bBjOnTuHXr16oVevXrh06VIZV05EVDFpPRhCQ0MxYsQIBAQEoFGj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",
      "text/plain": [
       "<Figure size 400x500 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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AAcwAOAA8+fPN5LML7/8kuN1vLy8TFhYWK72ExYWZry8vMzJkyfNyZMnzV9//WXeeOMN4+TkZOrVq2eMMebTTz81zs7O5scff7Rb94MPPjCSzObNm21tkoyrq6v566+/bG07d+40kszMmTNtbRcuXEhXS1RUlJFkPvnkE1vb999/bySZ77//3q7mwMDATI9p8+bNpnDhwubxxx+3tU2YMMF4eXmZP//8067vqFGjjIuLi4mLizPGGLNy5UojyUyZMsXW58qVK6ZFixZGkpk/f36m+7223qVLl2baZ9iwYUaS3flMTEw0FStWNEFBQSYlJcUYY8zUqVONJLNy5Upbv3///dfUqFEj23MydOhQ4+PjY65cuZJpHUuXLk23nTStWrUyrVq1sr2fN2+ekWSmTZuWrm9qamqm+0jbVo0aNWyfsb1795oXXnjBSDIPPPCArV9sbKxxcXExEydOtFt/165dplChQrb25ORkU6JECdOkSRNz+fJlW78FCxYYSXZ1p41HpUqV7D5zqamppmrVqiYkJMSu/gsXLpiKFSuadu3a2dp8fX3Nc889l+nxRUdHZzvmxhgTGBho9/OZ089B2jHUrFnTJCcn2/q+++67RpLZtWtXlvsFkD9w+xOA215SUpJKlSqlUqVKqUqVKnr55ZfVvHlzrVixQpK0dOlS1axZUzVq1NCpU6dsrzZt2khSur+cBwcH2/3lul69evLx8dHBgwdtbR4eHrb/vnz5sk6fPq0qVaqoaNGidreV5FZ8fLx69OihBg0a6P3337e1L126VC1atFCxYsXsjiE4OFgpKSnatGmTJGnNmjUqVKiQnn32Wdu6Li4uGjx48A3XdL01a9aoadOmuvfee21t3t7eeuqppxQbG6vdu3dLktatW6dy5cqpS5cutn7u7u4aMGBAtvsoWrSokpKStH79+jyp+csvv1TJkiUzPA85uf1u7969ts9YjRo19NZbb6lLly52t6ctX75cqampCg0NtRsjf39/Va1a1fY5+/XXX3X69GkNGDBAhQr9/w0Fffr0UbFixTLcf1hYmN1nLiYmRvv379cjjzyi06dP2/aVlJSktm3batOmTbbb4ooWLaqff/5Zf//9d4bbTrsS8c033+jChQvZnos0Of0cpOnfv7/dXJcWLVpIkt3PFYD8i9ufABR4//77r9293ZLk7+9v+293d3d9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+126dMm8+eabpnbt2sbNzc0UK1bMNGrUyIwfP96cO3fO1k9Sho/fvP5xmv/884/p37+/KVmypPH29jYhISFm79696frl5JGygYGBOToHiYmJZvTo0aZKlSrG1dXVlCxZ0tx9993m7bffNpcuXbL1O336tHnssceMj4+P8fX1NY899pjtsaE5faRsZq+0x4ceOHDA9OjRwxQtWtS4u7ubpk2bmq+//jrd9g4ePGgeeOAB4+HhYUqVKmWef/558+WXXxpJZuvWrZmek2XLlpn777/flC5d2ri6upoKFSqYp59+2hw7dsxu+3PmzDGVKlUyLi4uduf5+kfKGnP1cauvvPKKqVixoilcuLDx9/c3PXr0MAcOHMjynLRq1crUrl07w2UbN240kkx4eLit7csvvzT33nuv8fLyMl5eXqZGjRrmueeeM/v27bNbd8aMGSYwMNC4ubmZpk2bms2bN5tGjRqZ9u3b2/pk94jf6Oho89BDD5kSJUoYNzc3ExgYaEJDQ01kZKQx5urja1944QVTv359U6RIEePl5WXq169v3n//fds2Dh48aB5//HFTuXJl4+7ubooXL27uu+8+891339nt6/rPtjE5+xxkdgyHDh3K8c85AMdzMoYZUACA/GP69OkaPny4/ve//6lcuXKOLiffSE1NValSpfTQQw/l6BYxALiVuPYIAHCYf//91+79xYsX9eGHH6pq1ap3dKC4ePFiunkXn3zyic6cOaPWrVs7pigAyAJzKgAADvPQQw+pQoUKatCggc6dO6fPPvtMe/fu1aJFixxdmkNt3bpVw4cPV8+ePVWiRAnt2LFDH3/8serUqaOePXs6ujwASIdQAQBwmJCQEM2dO1eLFi1SSkqKatWqpYiICD388MOOLs2hgoKCFBAQoBkzZujMmTMqXry4+vbtq8mTJ9t96zQA5BfMqQAAAABgCXMqAAAAAFhCqAAAAABgyR03pyI1NVV///23ihQpkqMvEAIAAADuVMYYJSYmqmzZsll+aeUdFyr+/vtvBQQEOLoMAAAAoMA4cuSIypcvn+nyOy5UFClSRNLVE+Pj4+PgagAAAID8KyEhQQEBAbbfoTNzx4WKtFuefHx8CBUAAABADmQ3bYCJ2gAAAAAsIVQAAAAAsIRQAQAAAMCSO25OBQAAQEGWmpqqS5cuOboM3CYKFy4sFxcXy9shVAAA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nXrkkUeyvOJwIz799FOVKFFCoaGhqlOnju1Vv359dezY0TZhe9y4cZo6dapmzJih/fv3a8eOHbY5GK1atVLLli3VvXt3rV+/XocOHdLatWu1bt06SVLr1q118uRJTZkyRQcOHNCsWbO0du3abGurUKGCXF1dNXPmTB08eFCrVq3ShAkT7PoMGjRICQkJ6tWrl3799Vft379fn376qfbt22frExISIh8fH73++uvq379/Xp26TDk0VJw/f96WDKWrj/WKiYmx3S82evRo9e3b19b/mWee0cGDB/Xiiy9q7969ev/99/XFF19o+PDhjigfAAAAeWzatGkqVqyY7r77bnXu3FkhISFq2LBhnu5j3rx56tatW4Zfzte9e3etWrVKp06dUlhYmKZPn673339ftWvXVqdOnbR//35b3y+//FJNmjRR7969VatWLb344otKSUmRJNWsWVPvv/++Zs2apfr162vbtm1238uRmVKlSmnBggVaunSpatWqpcmTJ+vtt9+261OiRAlt2LBB58+fV6tWrdSoUSPNmTPH7hYwZ2dn9evXTykpKXa/T98sTub6m71uoY0bN+q+++5L1x4WFqYFCxaoX79+io2NtbsHbOPGjRo+fLh2796t8uXLa+zYsXaP2MpOQkKCfH19de7cOYdeteCe39sHY3l7YBxvH4wlblcXL160PZnI3d3d0eWgAHjiiSd08uTJbL+zI6vPVk5/d3boRO3WrVunm8ByrYy+Lbt169aKjo6+iVUBAAAABde5c+e0a9cuLV68OEdfApgXbvunPwEAAAB3kgcffFDbtm3TM888Y/cdIDcToQIAAAC4jdzsx8dmpEA9/QkAAABA/kOoAAAAAGAJoQIAAACAJYQKAAAAAJYQKgAAAABYQqgAAAAAYAmhAgAAAIAlfE8FAABAQTbO9xbv71yOuzo5OWW5PDw8XOPGjbuhMpycnLRixQp17do1R/2ffvppzZ07VxEREerZs+cN7ROZI1QAAADgpjh27Jjtv5csWaJXX31V+/bts7V5e3vfkjouXLigiIgIvfjii5o3b57DQ8WlS5fk6urq0BryGrc/AQAA4Kbw9/e3vXx9feXk5GTXFhERoZo1a8rd3V01atTQ+++/b1v30qVLGjRokMqUKSN3d3cFBgZq0qRJkqSgoCBJUrdu3eTk5GR7n5mlS5eqVq1aGjVqlDZt2qQjR47YLU9OTtZLL72kgIAAubm5qUqVKvr4449ty//44w916tRJPj4+KlKkiFq0aKEDBw5Iklq3bq1hw4bZba9r167q16+f7X1QUJAmTJigvn37ysfHR0899ZQk6aWXXlK1atXk6empSpUqaezYsbp8+bLdtr766is1adJE7u7uKlmypLp16yZJeu2111SnTp10x9qgQQONHTs2y/NxMxAqAAAAcMstWrRIr776qiZOnKg9e/bojTfe0NixY7Vw4UJJ0owZM7Rq1Sp98cUX2rdvnxYtWmQLD7/88oskaf78+Tp27JjtfWY+/vhjPfroo/L19VWHDh20YMECu+V9+/bV559/rhkzZmjPnj368MMPbVdRjh49qpYtW8rNzU0bNmzQ9u3b9fjjj+vKlSu5Ot63335b9evXV3R0tO2X/iJFimjBggXavXu33n33Xc2ZM0fvvPOObZ3Vq1erW7du6tixo6KjoxUZGammTZtKkh5//HHt2bPH7tijo6P122+/qX///rmqLS9w+xMAAABuufDwcE2dOlUPPfSQJKlixYravXu3PvzwQ4WFhSkuLk5Vq1bVvffeKycnJwUGBtrWLVWqlCSpaNGi8vf3z3I/+/fv19atW7V8+XJJ0qOPPqoRI0ZozJgxcnJy0p9//qkvvvhC69evV3BwsCSpUqVKtvVnzZolX19fRUREqHDhwpKkatWq5fp427Rpo+eff96ubcyYMbb/DgoK0siRI223aUnSxIkT1atXL40fP97Wr379+pKk8uXLKyQkRPPnz1eTJk0kXQ1ZrVq1sqv/VuFKBQAAAG6ppKQkHThwQE888YS8vb1tr9dff912W1G/fv0UExOj6tWra8iQIfr2229vaF/z5s1TSEiISpYsKUnq2LGjzp07pw0bNkiSYmJi5OLiolatWmW4fkxMjFq0aGELFDeqcePG6dqWLFmie+65R/7+/vL29taYMWMUFxdnt++2bdtmus0BAwbo888/18WLF3Xp0iUtXrxYjz/+uKU6bxRXKgAAAHBLnT9/XpI0Z84cNWvWzG6Zi4uLJKlhw4Y6dOiQ1q5dq++++06hoaEKDg7WsmXLcryflJQULVy4UPHx8SpUqJBd+7x589S2bVt5eHhkuY3sljs7O8sYY9d2/bwISfLy8rJ7HxUVpT59+mj8+PEKCQmxXQ2ZOnVqjvfduXNnubm5acWKFXJ1ddXly5fVo0ePLNe5WQgVAAAAuKX8/PxUtmxZHTx4UH369Mm0n4+Pjx5++GE9/PDD6tGjh9q3b68zZ86oePHiKly4sFJSUrLcz5o1a5SYmKjo6GhbWJGk33//Xf3799fZs2dVt25dpaam6ocffrDd/nStevXqaeHChbp8+XKGVytKlSpl95SrlJQU/f7777rvvvuyrG3Lli0KDAzUK6+8Yms7fPhwun1HRkZmOkeiUKFCCgsL0/z58+Xq6qpevXplG0RuFkIFAAAAbrnx48dryJAh8vX1Vfv27ZWcnKxff/1V//zzj0aMGKFp06apTJkyuuuuu+Ts7KylS5fK399fRYsWlXR1DkJkZKTuueceubm5qVixYun28fHHH+uBBx6wzUNIU6tWLQ0fPlyLFi3Sc889p7CwMD3++OOaMWOG6tevr8OHD+vEiRMKDQ3VoEGDNHPmTPXq1UujR4+Wr6+vtm7dqqZNm6p69epq06aNRowYodWrV6ty5cqaNm2azp49m+3xV61aVXFxcYqIiFCTJk20evVqrVixwq5PeHi42rZtq8qVK6tXr166cuWK1qxZo5deesnW58knn1TNmjUlSZs3b87lKOQd5lQAAADglnvyySc1d+5czZ8/X3Xr1lWrVq20YMECVaxYUdLVJyNNmTJFjRs3VpMmTRQbG6s1a9bI2fnqr69Tp07V+vXrFRAQoLvuuivd9o8fP67Vq1ere/fu6ZY5OzurW7dutsfGzp49Wz169NDAgQNVo0YNDRgwQElJSZKkEiVKaMOGDTp//rxatWqlRo0aac6cObarFo8//rjCwsLUt29f2yTp7K5SSFKXLl00fPhwDRo0SA0aNNCWLVvSPQq2devWWrp0qVatWqUGDRqoTZs22rZtm12fqlWr6u6771aNGjXS3Up2KzmZ628Cu80lJCTI19dX586dk4+Pj8PqCBq12mH7zonYyQ84uoQCg7G8PTCOtw/GErerixcv6tChQ6pYsaLc3d0dXQ7yCWOMqlatqoEDB2rEiBE3tI2sPls5/d2Z258AAACAAujkyZOKiIhQfHy8Q76b4lqECgAAAKAAKl26tEqWLKmPPvoowzkltxKhAgAAACiA8tMsBiZqAwAAALCEUAEAAADAEkIFAABAAZKfbnnB7SE1NdXyNphTAQAAUAAULlxYTk5OOnnypEqVKiUnJydHl4QCzhijS5cu6eTJk3J2dparq+sNb4tQAQAAUAC4uLiofPny+t///qfY2FhHl4PbiKenpypUqGD7YsEbQagAAAAoILy9vVW1alVdvnzZ0aXgNuHi4qJChQpZvvJFqAAAAChAXFxc5OLi4ugyADtM1AYAAABgCaECAAAAgCWECgAAAACWECoAAAAAWEKoAAAAAGAJoQIAAACAJYQKAAAAAJYQKgAAAABYQqgAAAAAYAmhAgAAAIAlhAoAAAAAlhAqAAAAAFhCqAAAAABgCaECAAAAgCWECgAAAACWECoAAAAAWEKoAAAAAGAJoQIAAACAJYQKAAAAAJYQKgAAAABYQqgAAAAAYAmhAgAAAIAlhAoAAAAAlhAqAAAAAFhCqAAAAABgCaECAAAAgCWECgAAAACWECoAAAAAWEKoAAAAAGAJoQIAAACAJYQKAAAAAJYQKgAAAABYQqgAAAAAYAmhAgAAAIAlhAoAAAAAlhAqAAAAAFji8FAxa9YsBQUFyd3dXc2aNdO2bduy7D99+nRVr15dHh4eCggI0PDhw3Xx4sVbVC0AAACA6zk0VCxZskQjRoxQeHi4duzYofr16yskJEQnTpzIsP/ixYs1atQohYeHa8+ePfr444+1ZMkSvfzyy7e4cgAAAABpHBoqpk2bpgEDBqh///6qVauWPvjgA3l6emrevHkZ9t+yZYvuuecePfLIIwoKCtL999+v3r17Z3t1AwAAAMDN47BQcenSJW3fvl3BwcH/X4yzs4KDgxUVFZXhOnfffbe2b99uCxEHDx7UmjVr1LFjx1tSMwAAAID0Cjlqx6dOnVJKSor8/Pzs2v38/LR3794M13nkkUd06tQp3XvvvTLG6MqVK3rmmWeyvP0pOTlZycnJtvcJCQl5cwAAAAAAJOWDidq5sXHjRr3xxht6//33tWPHDi1fvlyrV6/WhAkTMl1n0qRJ8vX1tb0CAgJuYcUAAADA7c9hVypKliwpFxcXHT9+3K79+PHj8vf3z3CdsWPH6rHHHtOTTz4pSapbt66SkpL01FNP6ZVXXpGzc/qMNHr0aI0YMcL2PiEhgWABAAAA5CGHXalwdXVVo0aNFBkZaWtLTU1VZGSkmjdvnuE6Fy5cSBccXFxcJEnGmAzXcXNzk4+Pj90LAAAAQN5x2JUKSRoxYoTCwsLUuHFjNW3aVNOnT1dSUpL69+8vSerbt6/KlSunSZMmSZI6d+6sadOm6a677lKzZs30119/aezYsercubMtXAAAAAC4tRwaKh5++GGdPHlSr776quLj49WgQQOtW7fONnk7Li7O7srEmDFj5OTkpDFjxujo0aMqVaqUOnfurIkTJzrqEAAAAIA7nkNDhSQNGjRIgwYNynDZxo0b7d4XKlRI4eHhCg8PvwWVAQAAAMiJAvX0JwAAAAD5D6ECAAAAgCWECgAAAACWECoAAAAAWEKoAAAAAGAJoQIAAACAJYQKAAAAAJYQKgAAAABYQqgAAAAAYAmhAgAAAIAlhAoAAAAAlhAqAAAAAFhCqAAAAABgCaECAAAAgCWECgAAAACWECoAAAAAWEKoAAAAAGAJoQIAAACAJYQKAAAAAJYQKgAAAABYQqgAAAAAYAmhAgAAAIAlhAoAAAAAlhAqAAAAAFhCqAAAAABgCaECAAAAgCWECgAAAACWFHJ0AQAAANcLGrXa0SVkK3byA44uAcg3uFIBAAAAwBJCBQAAAABLCBUAAAAALCFUAAAAALCEUAEAAADAEkIFAAAAAEsIFQAAAAAsIVQAAAAAsIRQAQAAAMASQgUAAAAASwgVAAAAACwhVAAAAACwhFABAAAAwBJCBQAAAABLCBUAAAAALCFUAAAAALCEUAEAAADAkkKOLgD51DhfR1eQtXHnHF0BANwY/v/19sFY3h4YxzzBlQoAAAAAlhAqAAAAAFhCqAAAAABgCaECAAAAgCWECgAAAACWECoAAAAAWEKoAAAAAGAJoQIAAACAJYQKAAAAAJYQKgAAAABYQqgAAAAAYAmhAgAAAIAlhAoAAAAAlhAqAAAAAFhCqAAAAABgCaECAAAAgCWECgAAAACWECoAAAAAWEKoAAAAAGAJoQIAAACAJQ4PFbNmzVJQUJDc3d3VrFkzbdu2Lcv+Z8+e1XPPPacyZcrIzc1N1apV05o1a25RtQAAAACuV8iRO1+yZIlGjBihDz74QM2aNdP06dMVEhKiffv2qXTp0un6X7p0Se3atVPp0qW1bNkylStXTocPH1bRokVvffEAAAAAJDk4VEybNk0DBgxQ//79JUkffPCBVq9erXnz5mnUqFHp+s+bN09nzpzRli1bVLhwYUlSUFDQrSwZAAAAwHUcdvvTpUuXtH37dgUHB/9/Mc7OCg4OVlRUVIbrrFq1Ss2bN9dzzz0nPz8/1alTR2+88YZSUlJuVdkAAAAArpPrUBEUFKTXXntNcXFxlnZ86tQppaSkyM/Pz67dz89P8fHxGa5z8OBBLVu2TCkpKVqzZo3Gjh2rqVOn6vXXX890P8nJyUpISLB7AQAAAMg7uQ4Vw4YN0/Lly1WpUiW1a9dOERERSk5Ovhm1pZOamqrSpUvro48+UqNGjfTwww/rlVde0QcffJDpOpMmTZKvr6/tFRAQcEtqBQAAAO4UNxQqYmJitG3bNtWsWVODBw9WmTJlNGjQIO3YsSPH2ylZsqRcXFx0/Phxu/bjx4/L398/w3XKlCmjatWqycXFxdZWs2ZNxcfH69KlSxmuM3r0aJ07d872OnLkSI5rBAAAAJC9G55T0bBhQ82YMUN///23wsPDNXfuXDVp0kQNGjTQvHnzZIzJcn1XV1c1atRIkZGRtrbU1FRFRkaqefPmGa5zzz336K+//lJqaqqt7c8//1SZMmXk6uqa4Tpubm7y8fGxewEAAADIOzccKi5fvqwvvvhCXbp00fPPP6/GjRtr7ty56t69u15++WX16dMn222MGDFCc+bM0cKFC7Vnzx49++yzSkpKsj0Nqm/fvho9erSt/7PPPqszZ85o6NCh+vPPP7V69Wq98cYbeu655270MAAAAABYlOtHyu7YsUPz58/X559/LmdnZ/Xt21fvvPOOatSoYevTrVs3NWnSJNttPfzwwzp58qReffVVxcfHq0GDBlq3bp1t8nZcXJycnf8/9wQEBOibb77R8OHDVa9ePZUrV05Dhw7VSy+9lNvDAAAAAJBHch0qmjRponbt2mn27Nnq2rWr7fsirlWxYkX16tUrR9sbNGiQBg0alOGyjRs3pmtr3ry5tm7dmquaAQAAANw8uQ4VBw8eVGBgYJZ9vLy8NH/+/BsuCgAAAEDBkes5FSdOnNDPP/+crv3nn3/Wr7/+midFAQAAACg4ch0qnnvuuQwfy3r06FEmTAMAAAB3oFyHit27d6thw4bp2u+66y7t3r07T4oCAAAAUHDkOlS4ubml+8I6STp27JgKFcr1FA0AAAAABVyuU8D999+v0aNH67///a98fX0lSWfPntXLL7+sdu3a5XmBAAAAKLiCRq12dAlZinV3dAW3h1yHirffflstW7ZUYGCg7rrrLklSTEyM/Pz89Omnn+Z5gQAAAADyt1yHinLlyum3337TokWLtHPnTnl4eKh///7q3bt3ht9ZAQAAAOD2dkOTILy8vPTUU0/ldS0AAAAACqAbnlm9e/duxcXF6dKlS3btXbp0sVwUAAAAgILjhr5Ru1u3btq1a5ecnJxkjJEkOTk5SZJSUlLytkIAAAAA+VquHyk7dOhQVaxYUSdOnJCnp6f++OMPbdq0SY0bN9bGjRtvQokAAAAA8rNcX6mIiorShg0bVLJkSTk7O8vZ2Vn33nuvJk2apCFDhig6Ovpm1AkAAAAgn8r1lYqUlBQVKVJEklSyZEn9/fffkqTAwEDt27cvb6sDAAAAkO/l+kpFnTp1tHPnTlWsWFHNmjXTlClT5Orqqo8++kiVKlW6GTUCAAAAyMdyHSrGjBmjpKQkSdJrr72mTp06qUWLFipRooSWLFmS5wUCAAAAyN9yHSpCQkJs/12lShXt3btXZ86cUbFixWxPgAIAAABw58jVnIrLly+rUKFC+v333+3aixcvTqAAAAAA7lC5ChWFCxdWhQoV+C4KAAAAADa5fvrTK6+8opdffllnzpy5GfUAAAAAKGByPafivffe019//aWyZcsqMDBQXl5edst37NiRZ8UBAAAAyP9yHSq6du16E8oAAAAAUFDlOlSEh4ffjDoAAAAAFFC5nlMBAAAAANfK9ZUKZ2fnLB8fy5OhAAAAgDtLrkPFihUr7N5fvnxZ0dHRWrhwocaPH59nhQEAAAAoGHIdKh588MF0bT169FDt2rW1ZMkSPfHEE3lSGAAAAICCIc/mVPznP/9RZGRkXm0OAAAAQAGRJ6Hi33//1YwZM1SuXLm82BwAAACAAiTXtz8VK1bMbqK2MUaJiYny9PTUZ599lqfFAQAAAMj/ch0q3nnnHbtQ4ezsrFKlSqlZs2YqVqxYnhYHAAAAIP/Ldajo16/fTSgDAAAAQEGV6zkV8+fP19KlS9O1L126VAsXLsyTogAAAAAUHLkOFZMmTVLJkiXTtZcuXVpvvPFGnhQFAAAAoODIdaiIi4tTxYoV07UHBgYqLi4uT4oCAAAAUHDkOlSULl1av/32W7r2nTt3qkSJEnlSFAAAAICCI9ehonfv3hoyZIi+//57paSkKCUlRRs2bNDQoUPVq1evm1EjAAAAgHws109/mjBhgmJjY9W2bVsVKnR19dTUVPXt25c5FQAAAMAdKNehwtXVVUuWLNHrr7+umJgYeXh4qG7dugoMDLwZ9QEAAADI53IdKtJUrVpVVatWzctaAAAAABRAuZ5T0b17d7355pvp2qdMmaKePXvmSVEAAAAACo5ch4pNmzapY8eO6do7dOigTZs25UlRAAAAAAqOXIeK8+fPy9XVNV174cKFlZCQkCdFAQAAACg4ch0q6tatqyVLlqRrj4iIUK1atfKkKAAAAAAFR64nao8dO1YPPfSQDhw4oDZt2kiSIiMjtXjxYi1btizPCwQAAACQv+U6VHTu3FkrV67UG2+8oWXLlsnDw0P169fXhg0bVLx48ZtRIwAAAIB87IYeKfvAAw/ogQcekCQlJCTo888/18iRI7V9+3alpKTkaYEAAAAA8rdcz6lIs2nTJoWFhals2bKaOnWq2rRpo61bt+ZlbQAAAAAKgFxdqYiPj9eCBQv08ccfKyEhQaGhoUpOTtbKlSuZpA0AAADcoXJ8paJz586qXr26fvvtN02fPl1///23Zs6ceTNrAwAAAFAA5PhKxdq1azVkyBA9++yzqlq16s2sCQAAAEABkuMrFT/99JMSExPVqFEjNWvWTO+9955OnTp1M2sDAAAAUADkOFT85z//0Zw5c3Ts2DE9/fTTioiIUNmyZZWamqr169crMTHxZtYJAAAAIJ/K9dOfvLy89Pjjj+unn37Srl279Pzzz2vy5MkqXbq0unTpcjNqBAAAAJCP3fAjZSWpevXqmjJliv73v//p888/z6uaAAAAABQglkJFGhcXF3Xt2lWrVq3Ki80BAAAAKEDyJFQAAAAAuHMRKgAAAABYQqgAAAAAYAmhAgAAAIAlhAoAAAAAlhAqAAAAAFhCqAAAAABgCaECAAAAgCWECgAAAACWECoAAAAAWJIvQsWsWbMUFBQkd3d3NWvWTNu2bcvRehEREXJyclLXrl1vboEAAAAAMuXwULFkyRKNGDFC4eHh2rFjh+rXr6+QkBCdOHEiy/ViY2M1cuRItWjR4hZVCgAAACAjDg8V06ZN04ABA9S/f3/VqlVLH3zwgTw9PTVv3rxM10lJSVGfPn00fvx4VapU6RZWCwAAAOB6Dg0Vly5d0vbt2xUcHGxrc3Z2VnBwsKKiojJd77XXXlPp0qX1xBNP3IoyAQAAAGShkCN3furUKaWkpMjPz8+u3c/PT3v37s1wnZ9++kkff/yxYmJicrSP5ORkJScn294nJCTccL0AAAAA0nP47U+5kZiYqMcee0xz5sxRyZIlc7TOpEmT5Ovra3sFBATc5CoBAACAO4tDr1SULFlSLi4uOn78uF378ePH5e/vn67/gQMHFBsbq86dO9vaUlNTJUmFChXSvn37VLlyZbt1Ro8erREjRtjeJyQkECwAAACAPOTQUOHq6qpGjRopMjLS9ljY1NRURUZGatCgQen616hRQ7t27bJrGzNmjBITE/Xuu+9mGBbc3Nzk5uZ2U+oHAAAA4OBQIUkjRoxQWFiYGjdurKZNm2r69OlKSkpS//79JUl9+/ZVuXLlNGnSJLm7u6tOnTp26xctWlSS0rUDAAAAuDUcHioefvhhnTx5Uq+++qri4+PVoEEDrVu3zjZ5Oy4uTs7OBWrqBwAAAHBHcXiokKRBgwZleLuTJG3cuDHLdRcsWJD3BQEAAADIMS4BAAAAALCEUAEAAADAEkIFAAAAAEsIFQAAAAAsIVQAAAAAsIRQAQAAAMASQgUAAAAASwgVAAAAACwhVAAAAACwhFABAAAAwBJCBQAAAABLCBUAAAAALCFUAAAAALCEUAEAAADAEkIFAAAAAEsIFQAAAAAsIVQAAAAAsIRQAQAAAMASQgUAAAAASwo5ugAAN9k4X0dXkLVx5xxdQcHAOAIA8jGuVAAAAACwhFABAAAAwBJCBQAAAABLCBUAAAAALCFUAAAAALCEUAEAAADAEkIFAAAAAEsIFQAAAAAsIVQAAAAAsIRQAQAAAMASQgUAAAAASwgVAAAAACwhVAAAAACwhFABAAAAwBJCBQAAAABLCBUAAAAALCFUAAAAALCEUAEAAADAEkIFAAAAAEsIFQAAAAAsIVQAAAAAsIRQAQAAAMASQgUAAAAASwgVAAAAACwhVAAAAACwhFABAAAAwBJCBQAAAABLCBUAAAAALCFUAAAAALCEUAEAAADAEkIFAAAAAEsIFQAAAAAsIVQAAAAAsIRQAQAAAMASQgUAAAAASwgVAAAAACwhVAAAAACwhFABAAAAwBJCBQAAAABLCBUAAAAALCFUAAAAALCEUAEAAADAEkIFAAAAAEsIFQAAAAAsIVQAAAAAsIRQAQAAAMCSfBEqZs2apaCgILm7u6tZs2batm1bpn3nzJmjFi1aqFixYipWrJiCg4Oz7A8AAADg5nJ4qFiyZIlGjBih8PBw7dixQ/Xr11dISIhOnDiRYf+NGzeqd+/e+v777xUVFaWAgADdf//9Onr06C2uHAAAAICUD0LFtGnTNGDAAPXv31+1atXSBx98IE9PT82bNy/D/osWLdLAgQPVoEED1ahRQ3PnzlVqaqoiIyNvceUAAAAAJAeHikuXLmn79u0KDg62tTk7Oys4OFhRUVE52saFCxd0+fJlFS9e/GaVCQAAACALhRy581OnTiklJUV+fn527X5+ftq7d2+OtvHSSy+pbNmydsHkWsnJyUpOTra9T0hIuPGCAQAAAKTj8NufrJg8ebIiIiK0YsUKubu7Z9hn0qRJ8vX1tb0CAgJucZUAAADA7c2hoaJkyZJycXHR8ePH7dqPHz8uf3//LNd9++23NXnyZH377beqV69epv1Gjx6tc+fO2V5HjhzJk9oBAAAAXOXQUOHq6qpGjRrZTbJOm3TdvHnzTNebMmWKJkyYoHXr1qlx48ZZ7sPNzU0+Pj52LwAAAAB5x6FzKiRpxIgRCgsLU+PGjdW0aVNNnz5dSUlJ6t+/vySpb9++KleunCZNmiRJevPNN/Xqq69q8eLFCgoKUnx8vCTJ29tb3t7eDjsOAAAA4E7l8FDx8MMP6+TJk3r11VcVHx+vBg0aaN26dbbJ23FxcXJ2/v8LKrNnz9alS5fUo0cPu+2Eh4dr3Lhxt7J0AAAAAMoHoUKSBg0apEGDBmW4bOPGjXbvY2Njb35BAAAAAHKsQD/9CQAAAIDjESoAAAAAWEKoAAAAAGAJoQIAAACAJYQKAAAAAJYQKgAAAABYQqgAAAAAYAmhAgAAAIAlhAoAAAAAlhAqAAAAAFhCqAAAAABgCaECAAAAgCWECgAAAACWECoAAAAAWEKoAAAAAGAJoQIAAACAJYQKAAAAAJYQKgAAAABYQqgAAAAAYAmhAgAAAIAlhAoAAAAAlhAqAAAAAFhCqAAAAABgCaECAAAAgCWECgAAAACWECoAAAAAWEKoAAAAAGAJoQIAAACAJYQKAAAAAJYQKgAAAABYQqgAAAAAYAmhAgAAAIAlhAoAAAAAlhAqAAAAAFhCqAAAAABgCaECAAAAgCWECgAAAACWECoAAAAAWEKoAAAAAGAJoQIAAACAJYQKAAAAAJYQKgAAAABYQqgAAAAAYAmhAgAAAIAlhAoAAAAAlhAqAAAAAFhCqAAAAABgCaECAAAAgCWECgAAAACWECoAAAAAWEKoAAAAAGAJoQIAAACAJYQKAAAAAJYQKgAAAABYQqgAAAAAYAmhAgAAAIAlhAoAAAAAlhAqAAAAAFhCqAAAAABgCaECAAAAgCWECgAAAACWECoAAAAAWEKoAAAAAGAJoQIAAACAJYQKAAAAAJYQKgAAAABYki9CxaxZsxQUFCR3d3c1a9ZM27Zty7L/0qVLVaNGDbm7u6tu3bpas2bNLaoUAAAAwPUcHiqWLFmiESNGKDw8XDt27FD9+vUVEhKiEydOZNh/y5Yt6t27t5544glFR0era9eu6tq1q37//fdbXDkAAAAAKR+EimnTpmnAgAHq37+/atWqpQ8++ECenp6aN29ehv3fffddtW/fXi+88IJq1qypCRMmqGHDhnrvvfduceUAAAAAJAeHikuXLmn79u0KDg62tTk7Oys4OFhRUVEZrhMVFWXXX5JCQkIy7Q8AAADg5irkyJ2fOnVKKSkp8vPzs2v38/PT3r17M1wnPj4+w/7x8fEZ9k9OTlZycrLt/blz5yRJCQkJVkq3LDX5gkP3n50EJ+PoErLm4PG7FmNpUT4ZS8bRonwyjhJjaVk+Gcv8Po4SY5lT+X0sGcfsdn91/8ZkfZ4cGipuhUmTJmn8+PHp2gMCAhxQTcHh6+gCsjM531eYb+T7M8VY5ki+P0uMY47l+zPFWOZYvj9TjGWO5PuzlE/GMTExUb6+mdfi0FBRsmRJubi46Pjx43btx48fl7+/f4br+Pv756r/6NGjNWLECNv71NRUnTlzRiVKlJCTk5PFI7g9JSQkKCAgQEeOHJGPj4+jy4EFjOXtgXG8fTCWtw/G8vbAOGbPGKPExESVLVs2y34ODRWurq5q1KiRIiMj1bVrV0lXf+mPjIzUoEGDMlynefPmioyM1LBhw2xt69evV/PmzTPs7+bmJjc3N7u2okWL5kX5tz0fHx9+wG4TjOXtgXG8fTCWtw/G8vbAOGYtqysUaRx++9OIESMUFhamxo0bq2nTppo+fbqSkpLUv39/SVLfvn1Vrlw5TZo0SZI0dOhQtWrVSlOnTtUDDzygiIgI/frrr/roo48ceRgAAADAHcvhoeLhhx/WyZMn9eqrryo+Pl4NGjTQunXrbJOx4+Li5Oz8/w+puvvuu7V48WKNGTNGL7/8sqpWraqVK1eqTp06jjoEAAAA4I7m8FAhSYMGDcr0dqeNGzema+vZs6d69ux5k6u6c7m5uSk8PDzdbWMoeBjL2wPjePtgLG8fjOXtgXHMO04mu+dDAQAAAEAWHP6N2gAAAAAKNkIFAAAAAEsIFQXEuHHj5O7urtDQUF25csXR5eAGMY63D8by9sFYFkyMW8HEuN3GDAqExMREs2HDBuPm5mYWL15saVupqalm7Nixxt/f37i7u5u2bduaP//8M9v13nvvPRMYGGjc3NxM06ZNzc8//2y3/N9//zUDBw40xYsXN15eXuahhx4y8fHxdn0GDx5sGjZsaFxdXU39+vUtHUdBVBDG8cMPPzStWrUyRYoUMZLMP//8k24bp0+fNo888ogpUqSI8fX1NY8//rhJTEy0dDwFjaPH8ocffjCdOnUyZcqUMZLMihUrbmi7jGXejuWXX35p2rVrZ4oXL24kmejo6Byt98UXX5jq1asbNzc3U6dOHbN69Wq75Yxlerd63HLyb9z18mrcdu7cae69917j5uZmypcvb958801Lx+tI+XHcDh8+bDp27Gg8PDxMqVKlzMiRI83ly5ez3HdejVt2P/sFCaGigAkLCzMdOnSwtI3JkycbX19fs3LlSrNz507TpUsXU7FiRfPvv/9muk5ERIRxdXU18+bNM3/88YcZMGCAKVq0qDl+/LitzzPPPGMCAgJMZGSk+fXXX81//vMfc/fdd9ttZ/Dgwea9994zjz322B0ZKtLk53F85513zKRJk8ykSZMyDRXt27c39evXN1u3bjU//vijqVKliundu7el4ymoHDWWa9asMa+88opZvnx5pqEiJ9tlLP9fXozlJ598YsaPH2/mzJmT41CxefNm4+LiYqZMmWJ2795txowZYwoXLmx27dpl68NYZu5WjVtO/o27Xl6M27lz54yfn5/p06eP+f33383nn39uPDw8zIcffmjpmB0tv4zblStXTJ06dUxwcLCJjo42a9asMSVLljSjR4/Oct95MW45+dkvSAgVBczs2bNNoUKFzIkTJ25o/dTUVOPv72/eeustW9vZs2eNm5ub+fzzzzNdr2nTpua5556zvU9JSTFly5Y1kyZNsm2jcOHCZunSpbY+e/bsMZJMVFRUuu2Fh4ff0aEiv47jtb7//vsMQ8Xu3buNJPPLL7/Y2tauXWucnJzM0aNHb+h4CjJHjeW1MgoVOdkuY2nP6lhe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tGjRdO2//PKLnnrqqTzZR1bmzJmj+vXry9vbW0WLFtVdd92lSZMm2Zbnx8/yjQbEWrVq6T//+Y+mTZuW90UByLcIFQBQgCxbtkw+Pj6655577Npr166tY8eOKS4uTvPnz9e6dev07LPPOqhKay5fvixXV1f5+/vLycnppu6rVKlS8vT0vKn7mDdvnoYNG6YhQ4YoJiZGmzdv1osvvqjz58/neluXL1++CRXmvf79+2v27Nm6cuWKo0sBcIsQKgDctqZNm6a6devKy8tLAQEBGjhwoN0vcml/vf76669VvXp1eXp6qkePHrpw4YIWLlyooKAgFStWTEOGDFFKSordthMTE9W7d295eXmpXLlymjVrlt3y/fv3q2XLlnJ3d1etWrW0fv36dPW99NJLqlatmjw9PVWpUiWNHTs2218aIyIi1Llz53TthQoVkr+/v8qVK6fg4GD17Nkz3T7nzp2rmjVryt3dXTVq1ND7779vt3zLli1q0KCB3N3d1bhxY61cuVJOTk6KiYmxO1/XSuuTmV9++UXt2rVTyZIl5evrq1atWmnHjh12fZycnDR79mx16dJFXl5emjhxYrrbn1q3bm27GnPtKzY2VlLWY71x40b1799f586ds603btw4Selvf4qLi9ODDz4ob29v+fj4KDQ0VMePH7ctT/vr/aeffqqgoCD5+vqqV69eSkxMzPQcrFq1SqGhoXriiSdUpUoV1a5dW71799bEiRNt21y4cKH++9//2urbuHGjYmNj5eTkpCVLlqhVq1Zyd3fXokWLsh3LtPWWL1+u++67T56enqpfv76ioqLs6pozZ44CAgLk6empbt26adq0abbxXbBggcaPH6+dO3faalqwYIFt3VOnTqlbt27y9PRU1apVtWrVKrttt2vXTmfOnNEPP/yQ6XkBcJsxAFBAhYWFmQcffDDT5e+8847ZsGGDOXTokImMjDTVq1c3zz77rG35/PnzTeHChU27du3Mjh07zA8//GBKlChh7r//fhMaGmr++OMP89VXXxlXV1cTERFhWy8wMNAUKVLETJo0yezbt8/MmDHDuLi4mG+//dYYY0xKSoqpU6eOadu2rYmJiTE//PCDueuuu4wks2LFCtt2JkyYYDZv3mwOHTpkVq1aZfz8/Mybb76Z5TH7+vra1WKMMeHh4aZ+/fq294cOHTK1a9c2fn5+trbPPvvMlClTxnz55Zfm4MGD5ssvvzTFixc3CxYsMMYYc+7cOVO8eHHz6KOPmj/++MOsWbPGVKtWzUgy0dHRtvPl6+trt+8VK1aYa/8pub6WyMhI8+mnn5o9e/aY3bt3myeeeML4+fmZhIQEWx9JpnTp0mbevHnmwIED5vDhw+b77783ksw///xjjDHm9OnT5tixY7bXQw89ZKpXr24uXLiQ7VgnJyeb6dOnGx8fH9v6iYmJtrF85513bOPWoEEDc++995pff/3VbN261TRq1Mi0atXK7vi8vb3NQw89ZHbt2mU2bdpk/P39zcsvv5zpmD399NOmRo0aJjY2NsPliYmJJjQ01LRv395WX3Jysjl06JCRZIKCgmzj9vfff2c7lmnr1ahRw3z99ddm3759pkePHiYwMNBcvnzZGGPMTz/9ZJydnc1bb71l9u3bZ2bNmmWKFy9uG98LFy6Y559/3tSuXdtWU9q5lmTKly9vFi9ebPbv32+GDBlivL29zenTp+2Oq1mzZiY8PDzT8wLg9kKoAFBgZRcqrrd06VJTokQJ2/v58+cbSeavv/6ytT399NPG09PT9kunMcaEhISYp59+2vY+MDDQtG/f3m7bDz/8sOnQoYMxxphvvvnGFCpUyBw9etS2fO3atelCxfXeeust06hRo0yX//PPP0aS2bRpk117eHi4cXZ2Nl5eXsbd3d1IMpLMtGnTbH0qV65sFi9ebLfehAkTTPPmzY0xxsyePduUKFHC/Pvvv7blc+bMsRwqrpeSkmKKFClivvrqK1ubJDNs2DC7fteHimtNmzbNFC1a1Ozbty/T/WQ01tfXbox9qPj222+Ni4uLiYuLsy3/448/jCSzbds22/F5enrahaIXXnjBNGvWLNNa/v77b/Of//zHSDLVqlUzYWFhZsmSJSYlJcXWJ6PPclo4mD59ul17dmOZtt7cuXPTHceePXuMMVc/rw888IDdNvr06WN3jjIbS0lmzJgxtvfnz583kszatWvt+nXr1s3069cvk7MC4HbD7U8Ablvfffed2rZtq3LlyqlIkSJ67LHHdPr0aV24cMHWx9PTU5UrV7a99/PzU1BQkLy9ve3aTpw4Ybft5s2bp3u/Z88eSdKePXsUEBCgsmXLZtpfkpYsWaJ77rlH/v7+8vb21pgxYxQXF5fp8fz777+SJHd393TLqlevrpiYGP3yyy966aWXFBISosGDB0uSkpKSdODAAT3xxBPy9va2vV5//XUdOHBAkrRv3z7Vq1fPbttNmzbNtJacOn78uAYMGKCqVavK19dXPj4+On/+fLrjbNy4cY62t3btWo0aNUpLlixRtWrVbO05GevspI1bQECAra1WrVoqWrSobWylq7dMFSlSxPa+TJky6T4f1ypTpoyioqK0a9cuDR06VFeuXFFYWJjat2+v1NTUbOu69tzkZCzT1KtXz64GSbY69+3bl258czPe127by8tLPj4+6c6Bh4dHrs4/gIKNUAHgthQbG6tOnTqpXr16+vLLL7V9+3bbvIdLly7Z+hUuXNhuPScnpwzbcvLLX25ERUWpT58+6tixo77++mtFR0frlVdesavteiVKlJCTk5P++eefdMtcXV1VpUoV1alTR5MnT5aLi4vGjx8vSba5BXPmzFFMTIzt9fvvv2vr1q05rtnZ2VnGGLu27OaAhIWFKSYmRu+++662bNmimJgYlShRIt1xenl5Zbv/3bt3q1evXpo8ebLdI3VzOtZ55UY/H3Xq1NHAgQP12Wefaf369Vq/fn2O5hxce25yM5bX1pk27yWvPsc5OQdnzpxRqVKl8mR/APK/Qo4uAABuhu3btys1NVVTp06Vs/PVv5988cUXebb963+B27p1q2rWrClJqlmzpo4cOaJjx47Z/kJ8ff8tW7YoMDBQr7zyiq3t8OHDWe7T1dVVtWrV0u7du7P9nooxY8aoTZs2evbZZ1W2bFmVLVtWBw8eVJ8+fTLsX716dX322WdKTk6Wm5ubpKuTrK9VqlQpJSYmKikpyfaLbtok7sxs3rxZ77//vjp27ChJOnLkiE6dOpXlOhk5deqUOnfurO7du2v48OF2y3Iy1q6urukm218vbdyOHDliu1qxe/dunT17VrVq1cp1zVlJ217adznkpD7p6lWz7MYyJ6pXr55ufK9/n9OaMvP777+rR48eN7w+gIKFUAGgQDt37ly6X2xLlCihKlWq6PLly5o5c6Y6d+6szZs364MPPsiz/W7evFlTpkxR165dtX79ei1dulSrV6+WJAUHB6tatWoKCwvTW2+9pYSEBLvwIElVq1ZVXFycIiIi1KRJE61evVorVqzIdr8hISH66aefNGzYsCz7NW/eXPXq1dMbb7yh9957T+PHj9eQIUPk6+ur9u3bKzk5Wb/++qv++ecfjRgxQo888oheeeUVPfXUUxo1apTi4uL09ttvS/r/v3I3a9ZMnp6eevnllzVkyBD9/PPPdk8EykjVqlX16aefqnHjxkpISNALL7wgDw+PbI/zet27d5enp6fGjRun+Ph4W3upUqVyNNZBQUE6f/68IiMjVb9+fXl6eqZ7lGxwcLDq1q2rPn36aPr06bpy5YoGDhyoVq1a5fj2rIykBbs2bdqofPnyOnbsmF5//XWVKlXKdltcUFCQvvnmG+3bt08lSpTI8vs5shvLnBg8eLBatmypadOmqXPnztqwYYPWrl1r9ySvoKAgHTp0SDExMSpfvryKFCliC5zZiY2N1dGjRxUcHJyj/gBuA46e1AEANyosLMw2Kfna1xNPPGGMuTqht0yZMsbDw8OEhISYTz75xG7yb0aTdzOanHr9JNrAwEAzfvx407NnT+Pp6Wn8/f3Nu+++a7fOvn37zL333mtcXV1NtWrVzLp169JN1H7hhRdMiRIljLe3t3n44YfNO++8k+Fk4mv98ccfxsPDw5w9ezbLmo0x5vPPPzdubm62iceLFi0yDRo0MK6urqZYsWKmZcuWZvny5bb+mzdvNvXq1TOurq6mUaNGZvHixUaS2bt3r63PihUrTJUqVYyHh4fp1KmT+eijj7KcqL1jxw7TuHFj4+7ubqpWrWqWLl1qNznaGJPhBPbrJ2pnNM6SzKFDh4wx2Y+1McY888wzpkSJEkaS7alE19dy+PBh06VLF+Pl5WWKFClievbsaeLj47M81++8844JDAxMd/7TLFu2zHTs2NGUKVPGuLq6mrJly5ru3bub3377zdbnxIkTpl27dsbb29tIMt9//71twnXaRPlrZTWWGa2XNsn/+++/t7V99NFHply5csbDw8N07drVvP7668bf39+2/OLFi6Z79+6maNGiRpKZP3++bSyuHy9fX1/bcmOMeeONN0xISEim5wTA7cfJmOtukAUA5Gs9e/ZUw4YNNXr06Ju6n0WLFtm+3+FGri6gYBkwYID27t2rH3/80dJ2Ll26pKpVq2rx4sXpvqQRwO2L258AoIB566239NVXX+X5dj/55BNVqlRJ5cqV086dO/XSSy8pNDSUQHGbevvtt9WuXTt5eXlp7dq1WrhwYbovRLwRcXFxevnllwkUwB2GKxUAAEnSlClT9P777ys+Pl5lypRR165dNXHixHRzD3B7CA0N1caNG5WYmKhKlSpp8ODBeuaZZxxdFoACilABAAAAwBK+pwIAAACAJYQKAAAAAJYQKgAAAABYQqgAAAAAYAmhAgAAAIAlhAoAAAAAlhAqAAAAAFhCqAAAAABgCaECAAAAgCX/B9j+naCJ154nAAAAAElFTkSuQmCC",
      "text/plain": [
       "<Figure size 800x500 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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evRoVaxYMUdjWaFCBW3dulWXL1/OldvX1q5dW0888YQ++OADDRs2TGXKlFGFChVkjFG5cuWyDHlpx+evv/7SAw88YGu/cuWKYmJiVLt27Wx9Hkny9fW97ndCkkqUKKF+/fqpX79+OnHihO6++26NHz/eFiokqVatWqpVq5ZGjRqlzZs367777tPs2bP1xhtvZPo5svM9AHB7YE4FgDzt7bff1uTJkzVy5EgNGjQowz4lSpRQSEiI3SsnunXrpiNHjmjOnDnpll28eFGJiYk5rtvFxSXdX1hnzJihlJSUHG/r999/1zPPPKMnnngi02PQrVs3RUVF6Ztvvkm37OzZs7py5YokqW3btrpy5YpmzZplW56SkqIZM2bkuK7MtG3bVtu2bVNUVJStLTExUR9++KGCg4NVvXp1Sf+FwSNHjmjlypW2fpcuXcpwHK51+vRpu/fOzs62X7bTbgPs5eUlSdkKcp07d9apU6f03nvvpVt2o38pf/nll3X58mXbGbBHHnlELi4uGjt2bLptGmNsn6l+/foqWrSo5syZYxs3SVq4cGG2LwWqV6+eKlSooMmTJ+v8+fPplp88eVLSf2N/7eV4xYsXV8mSJW3HMT4+3q4O6b+A4ezsnOVT6bP7PQBwe+BMBQCHmjt3brpnPEjSoEGD9N133+nll19WpUqVVK1aNS1YsMCuT6tWrRQQEGC5hieffFJffPGF+vbtq++//1733XefUlJStHfvXn3xxRf65ptvVL9+/Rxts127dvrss8/k5+en6tWr2y5bSrvlbE6kXdrVtGnTdMfg3nvvVfny5fXSSy9p5cqVateunXr16qV69eopMTFRu3bt0tKlSxUTEyN/f3+1b99e9913n4YPH66YmBhVr15dy5Yty/E8jy+//NL2F+erhYWFafjw4fr888/Vpk0bDRw4UEWKFNEnn3yigwcP6ssvv7Q9lO25557Te++9px49emjQoEEqUaKEFi5caJuHkNWZhmeeeUZnzpxRixYtVLp0aR06dEgzZsxQ3bp1bdfs161bVy4uLnrrrbd07tw5ubm5qUWLFrbb8F6tZ8+e+vTTTzV06FBt27ZNTZo0UWJior777jv169dPDz/8cI6OjyRVr15dbdu21UcffaTRo0erQoUKeuONNzRixAjbLWJ9fHx08OBBLV++XM8++6yGDRsmV1dXjRkzRgMGDFCLFi3UrVs3xcTEaP78+RnOQcmIs7OzPvroI7Vp00Y1atRQ7969VapUKR05ckTff/+9fH199dVXXykhIUGlS5dWly5dVKdOHXl7e+u7777Tzz//bDvLtn79evXv319du3ZV5cqVdeXKFX322WdycXFR586dM60hu98DALcJB911CsAdLu2Wspm9Dh8+bMLDw7Psk9GtQq+VdkvZ60lOTjZvvfWWqVGjhnFzczOFCxc29erVM2PHjjXnzp2z9ZOU4e03r72d5r///mt69+5t/P39jbe3twkNDTV79+5N1y87t5QtW7Zspsfg6lvAJiQkmBEjRpiKFSsaV1dX4+/vb+69914zefJkk5ycbOt3+vRp8+STTxpfX1/j5+dnnnzySdttQ7N7S9nMXmm3Dz1w4IDp0qWLKVSokHF3dzcNGzY0X3/9dbrt/f333+ahhx4yHh4eplixYubFF180X375pZFktmzZkukxWbp0qXnwwQdN8eLFjaurqylTpox57rnnzLFjx+y2P2fOHFO+fHnj4uJid5yvvaWsMf/dbvXVV1815cqVMwULFjSBgYGmS5cu5sCBA1kek2bNmpkaNWpkuGzDhg1GkgkPD7e1ffnll+b+++83Xl5exsvLy1StWtW88MILZt++fXbrTp8+3ZQtW9a4ubmZhg0bmk2bNpl69eqZ1q1b2/pc7xa/O3fuNI888ogpWrSocXNzM2XLljXdunUzkZGRxpj/bl/70ksvmTp16hgfHx/j5eVl6tSpY95//33bNv7++2/z1FNPmQoVKhh3d3dTpEgR88ADD5jvvvvObl/XfreNyd73ILPPcPDgwWx9JwHkDU7GMAMKAJB3TJs2TUOGDNE///yjUqVKObqcPCM1NVXFihXTI488kq1LxADgVuLcIwDAYS5evGj3/tKlS/rggw9UqVKlOzpQXLp0Kd28i08//VRnzpxR8+bNHVMUAGSBORUAAId55JFHVKZMGdWtW1fnzp3TggULtHfvXi1cuNDRpTnUli1bNGTIEHXt2lVFixbVjh079PHHH6tmzZrq2rWro8sDgHQIFQAAhwkNDdVHH32khQsXKiUlRdWrV1dERIQeffRRR5fmUMHBwQoKCtL06dN15swZFSlSRD179tTEiRPtnjoNAHkFcyoAAAAAWMKcCgAAAACWECoAAAAAWHLHzalITU3V0aNH5ePjk60HCAEAAAB3KmOMEhISVLJkySwfWnnHhYqjR48qKCjI0WUAAAAA+cbhw4dVunTpTJffcaHCx8dH0n8HxtfX18HVAAAAAHlXfHy8goKCbL9DZ+aOCxVplzz5+voSKgAAAIBsuN60ASZqAwAAALCEUAEAAADAEkIFAAAAAEvuuDkVAAAA+VlqaqqSk5MdXQZuEwULFpSLi4vl7RAqAAAA8onk5GQdPHhQqampji4Ft5FChQopMDDQ0jPcCBUAAAD5gDFGx44dk4uLi4KCgrJ8EBmQHcYYXbhwQSdOnJAklShR4oa3RagAAADIB65cuaILFy6oZMmS8vT0dHQ5uE14eHhIkk6cOKHixYvf8KVQRFwAAIB8ICUlRZLk6urq4Epwu0kLqZcvX77hbRAqAAAA8hEr170DGcmN7xShAgAAAIAlhAoAAADkK8HBwZo2bZqjy8BVmKgNAACQjwUPX3VL9xcz8aFs973eZTXh4eEaM2ZMjmv4+eef5eXlleP1MvL555/riSeeUN++fTVz5sxc2eadyKFnKn788Ue1b99eJUuWlJOTk1asWHHddTZs2KC7775bbm5uqlixoubPn3/T6wQAAEDOHTt2zPaaNm2afH197dqGDRtm62uM0ZUrV7K13WLFiuXaHbA+/vhjvfzyy/r888916dKlXNnmjcrPDzV0aKhITExUnTp1sp0KDx48qIceekgPPPCAoqOjNXjwYD3zzDP65ptvbnKlAAAAyKnAwEDby8/PT05OTrb3e/fulY+Pj9asWaN69erJzc1NGzdu1IEDB/Twww8rICBA3t7eatCggb777ju77V57+ZOTk5M++ugjderUSZ6enqpUqZJWrlx53foOHjyozZs3a/jw4apcubKWLVuWrs/cuXNVo0YNubm5qUSJEurfv79t2dmzZ/Xcc88pICBA7u7uqlmzpr7++mtJ0pgxY1S3bl27bU2bNk3BwcG297169VLHjh01fvx4lSxZUlWqVJEkffbZZ6pfv758fHwUGBioxx57zPYsiTR//PGH2rVrJ19fX/n4+KhJkyY6cOCAfvzxRxUsWFBxcXF2/QcPHqwmTZpc95jcKIeGijZt2uiNN95Qp06dstV/9uzZKleunKZMmaJq1aqpf//+6tKli955552bXCkAAABuhuHDh2vixInas2ePateurfPnz6tt27aKjIzUzp07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ePFiNWvWTO7u7lq4cOF1xzJtvWXLlumBBx6Qp6en6tSpo6ioKLu65syZo6CgIHl6eqpTp06aOnWqbXznz5+vsWPH6tdff7XVNH/+fNu6p06dUqdOneTp6alKlSpp5cqVdttu1aqVzpw5ox9++CHT4wLgNmMAIJ8KCwszDz/8cKbL33nnHbN+/Xpz8OBBExkZaapUqWKef/552/J58+aZggULmlatWpkdO3aYH374wRQtWtQ8+OCDplu3buaPP/4wX331lXF1dTURERG29cqWLWt8fHzMhAkTzL59+8z06dONi4uL+fbbb40xxqSkpJiaNWuali1bmujoaPPDDz+Yu+66y0gyy5cvt21n3LhxZtOmTebgwYNm5cqVJiAgwLz11ltZfmY/Pz+7WowxJjw83NSpU8f2/uDBg6ZGjRomICDA1rZgwQJTokQJ8+WXX5q///7bfPnll6ZIkSJm/vz5xhhjzp07Z4oUKWKeeOIJ88cff5jVq1ebypUrG0lm586dtuPl5+dnt+/ly5ebq/8pubaWyMhI89lnn5k9e/aY3bt3m6efftoEBASY+Ph4Wx9Jpnjx4mbu3LnmwIED5tChQ+b77783ksy///5rjDHm9OnT5tixY7bXI488YqpUqWIuXLhw3bFOSkoy06ZNM76+vrb1ExISbGP5zjvv2Matbt265v777ze//PKL2bJli6lXr55p1qyZ3efz9vY2jzzyiNm1a5f58ccfTWBgoBk5cmSmY/bcc8+ZqlWrmpiYmAyXJyQkmG7dupnWrVvb6ktKSjIHDx40kkxwcLBt3I4ePXrdsUxbr2rVqubrr782+/btM126dDFly5Y1ly9fNsYYs3HjRuPs7Gzefvtts2/fPjNz5kxTpEgR2/heuHDBvPjii6ZGjRq2mtKOtSRTunRps2jRIrN//34zcOBA4+3tbU6fPm33uRo1amTCw8MzPS4Abi+ECgD51vVCxbWWLFliihYtans/b948I8n89ddftrbnnnvOeHp62n7pNMaY0NBQ89xzz9nely1b1rRu3dpu248++qhp06aNMcaYb775xhQoUMAcOXLEtnzNmjXpQsW13n77bVOvXr1Ml//7779Gkvnxxx/t2sPDw42zs7Px8vIy7u7uRpKRZKZOnWrrU6FCBbNo0SK79caNG2caN25sjDFm1qxZpmjRoubixYu25XPmzLEc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",
      "text/plain": [
       "<Figure size 800x500 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "\n",
    "# Store accuracy values \n",
    "# Unpenalized model\n",
    "unpenalized_train_acc = train_metrics['Accuracy']\n",
    "unpenalized_test_acc = test_metrics['Accuracy']\n",
    "\n",
    "# L1 model accuracies\n",
    "l1_train_accuracies = []\n",
    "l1_test_accuracies = []\n",
    "l1_lambdas = []\n",
    "\n",
    "for lambda_value, model in models_L1.items():\n",
    "    l1_lambdas.append(lambda_value)\n",
    "    train_metrics = evaluate_model(model, X_train, Y_train)\n",
    "    test_metrics = evaluate_model(model, X_test, Y_test)\n",
    "    l1_train_accuracies.append(train_metrics['Accuracy'])\n",
    "    l1_test_accuracies.append(test_metrics['Accuracy'])\n",
    "\n",
    "# L2 model accuracies\n",
    "l2_train_accuracies = []\n",
    "l2_test_accuracies = []\n",
    "l2_lambdas = []\n",
    "\n",
    "for lambda_value, model in models_L2.items():\n",
    "    l2_lambdas.append(lambda_value)\n",
    "    train_metrics = evaluate_model(model, X_train, Y_train)\n",
    "    test_metrics = evaluate_model(model, X_test, Y_test)\n",
    "    l2_train_accuracies.append(train_metrics['Accuracy'])\n",
    "    l2_test_accuracies.append(test_metrics['Accuracy'])\n",
    "\n",
    "# Plotting \n",
    "\n",
    "def plot_accuracy_barplot(train_acc, test_acc, lambdas, title):\n",
    "    bar_width = 0.35\n",
    "    x = range(len(lambdas))\n",
    "\n",
    "    plt.figure(figsize=(8, 5))\n",
    "    plt.bar([i - bar_width/2 for i in x], train_acc, width=bar_width, label='Train Accuracy')\n",
    "    plt.bar([i + bar_width/2 for i in x], test_acc, width=bar_width, label='Test Accuracy')\n",
    "\n",
    "    plt.xlabel('Lambda (Regularization Strength)')\n",
    "    plt.ylabel('Accuracy')\n",
    "    plt.title(title)\n",
    "    plt.xticks(ticks=x, labels=[str(l) for l in lambdas])\n",
    "    plt.legend()\n",
    "    plt.tight_layout()\n",
    "    plt.show()\n",
    "\n",
    "# Unpenalized (single bar plot)\n",
    "plt.figure(figsize=(4, 5))\n",
    "plt.bar(['Train'], [unpenalized_train_acc], width=0.4, label='Train Accuracy')\n",
    "plt.bar(['Test'], [unpenalized_test_acc], width=0.4, label='Test Accuracy')\n",
    "plt.ylabel('Accuracy')\n",
    "plt.title('Unpenalized Logistic Regression')\n",
    "plt.legend()\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "\n",
    "# L1 barplot\n",
    "plot_accuracy_barplot(l1_train_accuracies, l1_test_accuracies, l1_lambdas, 'L1-Penalized Logistic Regression')\n",
    "\n",
    "# L2 barplot\n",
    "plot_accuracy_barplot(l2_train_accuracies, l2_test_accuracies, l2_lambdas, 'L2-Penalized Logistic Regression')\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "95f47274",
   "metadata": {},
   "source": [
    "Which method in combination with which parameter gives the best results on the test set?\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c5b0fe7a",
   "metadata": {},
   "source": [
    "L1 Penalized Logistic Regression Model at lambda = 10.000 give good 'Accuracy': 0.7142857142857143 on trained set and 'Accuracy': 0.6666666666666666 on test set. Therefore, this can be generalized well. "
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": ".venv",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
