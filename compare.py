import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import time
import os

# 1. 仅加载训练集 (a9a.txt)
def load_a9a_train():
    """加载训练集并预处理"""
    X, y = load_svmlight_file(os.path.join('a9a', 'a9a.txt'))
    X = X.toarray()
    y = np.where(y == -1, 0, 1)  # 转换标签为{0,1}
    return X, y

X, y = load_a9a_train()

# 数据标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 定义逻辑回归损失函数和梯度
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_loss(w, X, y):
    z = np.dot(X, w)
    y = y.reshape(-1, 1)
    return np.mean(-y * z + np.log(1 + np.exp(z)))

def logistic_gradient(w, X, y):
    y = y.reshape(-1, 1)
    return np.dot(X.T, (sigmoid(np.dot(X, w)) - y)) / len(y)

# 优化算法实现 (与之前相同)
def sgd(X, y, learning_rate=0.00001, n_epochs=50, batch_size=32):
    n_samples, n_features = X.shape
    w = np.zeros((n_features, 1))
    losses = []
    accuracies = []
    times = []
    start_time = time.time()
    
    for epoch in range(n_epochs):
        for i in range(0, n_samples, batch_size):
            X_batch = X[i:i+batch_size]
            y_batch = y[i:i+batch_size]
            grad = logistic_gradient(w, X_batch, y_batch)
            w -= learning_rate * grad
        
        loss = logistic_loss(w, X, y)
        y_pred = sigmoid(np.dot(X, w)) > 0.5
        acc = accuracy_score(y, y_pred)
        
        losses.append(loss)
        accuracies.append(acc)
        times.append(time.time() - start_time)
    
    return w, losses, accuracies, times

def sag(X, y, learning_rate=0.00001, n_epochs=50):
    n_samples, n_features = X.shape
    w = np.zeros((n_features, 1))
    losses = []
    accuracies = []
    times = []
    start_time = time.time()
    
    gradient_memory = np.zeros((n_samples, n_features, 1))
    sum_gradients = np.zeros((n_features, 1))
    
    for epoch in range(n_epochs):
        for i in np.random.permutation(n_samples):
            old_gradient = gradient_memory[i]
            new_gradient = logistic_gradient(w, X[i:i+1], y[i:i+1])
            sum_gradients += new_gradient - old_gradient
            gradient_memory[i] = new_gradient
            w -= learning_rate * sum_gradients / n_samples
        
        loss = logistic_loss(w, X, y)
        y_pred = sigmoid(np.dot(X, w)) > 0.5
        acc = accuracy_score(y, y_pred)
        
        losses.append(loss)
        accuracies.append(acc)
        times.append(time.time() - start_time)
    
    return w, losses, accuracies, times

def saga(X, y, learning_rate=0.00001, n_epochs=50):
    n_samples, n_features = X.shape
    w = np.zeros((n_features, 1))
    losses = []
    accuracies = []
    times = []
    start_time = time.time()
    
    gradient_memory = np.zeros((n_samples, n_features, 1))
    sum_gradients = np.zeros((n_features, 1))
    avg_gradient = np.zeros((n_features, 1))
    
    for epoch in range(n_epochs):
        for i in np.random.permutation(n_samples):
            old_gradient = gradient_memory[i]
            new_gradient = logistic_gradient(w, X[i:i+1], y[i:i+1])
            sum_gradients += new_gradient - old_gradient
            gradient_memory[i] = new_gradient
            avg_gradient = sum_gradients / n_samples
            w -= learning_rate * (new_gradient - old_gradient + avg_gradient)
        
        loss = logistic_loss(w, X, y)
        y_pred = sigmoid(np.dot(X, w)) > 0.5
        acc = accuracy_score(y, y_pred)
        
        losses.append(loss)
        accuracies.append(acc)
        times.append(time.time() - start_time)
    
    return w, losses, accuracies, times

def svrg(X, y, learning_rate=0.00001, n_epochs=50, m=None):
    n_samples, n_features = X.shape
    w = np.zeros((n_features, 1))
    losses = []
    accuracies = []
    times = []
    start_time = time.time()
    
    if m is None:
        m = n_samples
    
    for epoch in range(n_epochs):
        w_old = w.copy()
        full_gradient = logistic_gradient(w_old, X, y)
        
        for _ in range(m):
            i = np.random.randint(n_samples)
            grad_i = logistic_gradient(w, X[i:i+1], y[i:i+1])
            grad_i_old = logistic_gradient(w_old, X[i:i+1], y[i:i+1])
            w -= learning_rate * (grad_i - grad_i_old + full_gradient)
        
        loss = logistic_loss(w, X, y)
        y_pred = sigmoid(np.dot(X, w)) > 0.5
        acc = accuracy_score(y, y_pred)
        
        losses.append(loss)
        accuracies.append(acc)
        times.append(time.time() - start_time)
    
    return w, losses, accuracies, times

# 运行所有算法
n_epochs = 30
print("开始运行优化算法...")
algorithms = {
    'SGD': sgd(X, y, n_epochs=n_epochs),
    'SAG': sag(X, y, n_epochs=n_epochs),
    'SAGA': saga(X, y, n_epochs=n_epochs),
    'SVRG': svrg(X, y, n_epochs=n_epochs)
}
print("算法运行完成!")

# 绘制结果
plt.figure(figsize=(15, 10))

# 按epoch绘制的loss曲线
plt.subplot(2, 2, 1)
for name, (_, losses, _, _) in algorithms.items():
    plt.plot(range(1, n_epochs+1), losses, label=name)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs Epoch')
plt.legend()

# 按epoch绘制的accuracy曲线
plt.subplot(2, 2, 2)
for name, (_, _, accuracies, _) in algorithms.items():
    plt.plot(range(1, n_epochs+1), accuracies, label=name)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Epoch')
plt.legend()


plt.tight_layout()
plt.show()