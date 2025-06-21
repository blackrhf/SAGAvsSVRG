import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import os

# 1. 加载数据
def load_data():
    """加载并预处理数据"""
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

# 3. SVRG算法实现
def svrg(X, y, learning_rate=0.01, n_epochs=50, m=None):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    losses = []
    accuracies = []
    
    # 设置内循环迭代次数（默认为样本数）
    if m is None:
        m = n_samples
    
    for epoch in range(n_epochs):
        # 保存当前权重和全量梯度
        w_old = w.copy()
        full_gradient = logistic_gradient(w_old, X, y)
        
        # 内循环
        for _ in range(m):
            # 随机选择一个样本
            i = np.random.randint(n_samples)
            x_i = X[i:i+1]
            y_i = y[i:i+1]
            
            # 计算当前梯度和锚点梯度
            grad_i = logistic_gradient(w, x_i, y_i)
            grad_i_old = logistic_gradient(w_old, x_i, y_i)
            
            # SVRG关键更新步骤
            w -= learning_rate * (grad_i - grad_i_old + full_gradient)
        
        # 记录当前性能
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
print("开始训练SVRG...")
w, losses, accuracies = svrg(X, y, learning_rate=0.001, n_epochs=300, m=100)
print("训练完成!")

# 5. 绘制曲线
plt.figure(figsize=(12, 5))

# Loss曲线
plt.subplot(1, 2, 1)
plt.plot(losses, label='SVRG Loss', color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('SVRG: Loss over Epochs')
plt.grid(True)

# Accuracy曲线
plt.subplot(1, 2, 2)
plt.plot(accuracies, label='SVRG Accuracy', color='blue')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('SVRG: Accuracy over Epochs')
plt.grid(True)

plt.tight_layout()
plt.show()