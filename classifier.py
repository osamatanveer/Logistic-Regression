import numpy as np
import pandas as pd

df_X = pd.read_csv("question-3-features-train.csv")
amount_arr = df_X["Amount"].values
amount_arr = (amount_arr - np.mean(amount_arr)) / np.std(amount_arr)
df_X["Amount"] = amount_arr
X = df_X.to_numpy()

df_Y = pd.read_csv("question-3-labels-train.csv")
Y = df_Y.to_numpy()

df_X_test = pd.read_csv("question-3-features-test.csv")
df_X_test["Amount"] -= np.mean(df_X["Amount"])
df_X_test["Amount"] /= np.std(df_X["Amount"])
test_X = df_X_test.to_numpy()
df_Y_test = pd.read_csv("question-3-labels-test.csv")

def runGradientAscent(w, X, Y, alpha, batch_size, iters=1000):
  X_train = X
  Y_train = Y
  for i in range(iters):
    startLocation = 0 # start position of data
    if (batch_size < X.shape[0]):
      data = np.c_[X_train, Y_train]
      np.random.shuffle(data)
      X_train = data[:, :-1]
      Y_train = data[:, -1]
    for j in range(int(X.shape[0] / batch_size)):
      x = X_train[startLocation : startLocation + batch_size]
      y = Y_train[startLocation:startLocation + batch_size]
      Z = np.dot(x, w.T)
      A = 1.0 / (1.0 + np.exp(-Z))
      dA = y - A.T
      w = w + alpha * np.dot(dA, x) 
      startLocation = (startLocation + batch_size) % X.shape[0]
  return w

def trainLogisticRegressionModel(X,y, batch_size=1, alpha=0.0001):
  w = np.random.normal(0, 0.01, (1, X.shape[1]))
  w = runGradientAscent(w, X, y, alpha, batch_size)
  return w
alpha = 0.00001
w = trainLogisticRegressionModel(X, Y.flatten(), batch_size=100, alpha=alpha)

def testLogisticRegressionModel(w, X, y):
  predictions = ((np.dot(w, X.T) >= 0.5).astype(int)).flatten()
  print("Accuracy: ", np.mean(predictions == y) * 100)
  TP = sum(predictions[np.where(predictions == y)] == 1)
  TN = sum(predictions[np.where(predictions == y)] == 0)
  FP = sum(predictions[np.where(predictions != y)] == 1)
  FN = sum(predictions[np.where(predictions != y)] == 0)
  print("Confusion Matrix")
  print("TP: ", TP, "FP: ", FP)
  print("FN: ", FN, "TN: ", TN)
  precision = TP / (TP + FP)
  print("Precision: ", precision)
  recall = TP / (TP + FN)
  print("Recall: ", recall)
  NPV = TN / (FN + TN)
  print("NPV: ", NPV)
  FPR = FP / (TP + FP)
  print("FPR: ", FPR)
  FDR = FP / (FN + TN)
  print("FDR: ", FDR)
  F1 = 2 * precision * recall / (precision + recall)
  print("F1: ", F1)
  F2 = 5 * precision * recall / ((4 * precision) + recall)
  print("F2: ", F2)
testLogisticRegressionModel(w, test_X, df_Y_test.to_numpy().flatten())