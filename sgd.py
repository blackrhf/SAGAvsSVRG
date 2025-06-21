import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import os

# 1. 加载数据
def load_data():
    """加载数据并预处理"""
    X, y = load_svmlight_file(os.path.join('a9a', 'a9a.txt'))
    X = X.toarray()
    y = np.where(y == -1, 0, 1)  # 转换标签为{0,1}
    return X, y

X, y = load_data()

# 数据标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 2. 定义逻辑回归函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_loss(w, X, y):
    z = np.dot(X, w)
    return np.mean(-y * z + np.log(1 + np.exp(z)))

def logistic_gradient(w, X, y):
    return np.dot(X.T, (sigmoid(np.dot(X, w)) - y)) / len(y)

# 3. SGD实现
def sgd(X, y, learning_rate=0.01, n_epochs=5000, batch_size=32):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    losses = []
    accuracies = []
    
    for epoch in range(n_epochs):
        # 随机打乱数据
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        # 小批量梯度下降
        for i in range(0, n_samples, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
            grad = logistic_gradient(w, X_batch, y_batch)
            w -= learning_rate * grad
        
        # 计算当前loss和accuracy
        loss = logistic_loss(w, X, y)
        y_pred = sigmoid(np.dot(X, w)) > 0.5
        acc = accuracy_score(y, y_pred)
        
        losses.append(loss)
        accuracies.append(acc)
        
        # 打印进度
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {loss:.4f}, Accuracy: {acc:.4f}")
    
    return w, losses, accuracies

# 4. 训练模型
print("开始训练...")
w, losses, accuracies = sgd(X, y, learning_rate=0.00001, n_epochs=1000)
print("训练完成!")

# 5. 绘制曲线
plt.figure(figsize=(12, 5))

# Loss曲线
plt.subplot(1, 2, 1)
plt.plot(losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()

# Accuracy曲线
plt.subplot(1, 2, 2)
plt.plot(accuracies, label='Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy over Epochs')
plt.legend()

plt.tight_layout()
plt.show()