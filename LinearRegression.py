import pandas as pd
import matplotlib.pyplot as plt

def loss_function(m,b,points):
    total_error =0
    for i in range(len(points)):
        x = points.iloc[i].Age
        y = points.iloc[i].Premium
        total_error += (y - (m*x + b)) ** 2
    return total_error / float(len(points))
def gradient_descent(m,b, points, learning_rate):
    m_gradient = 0
    b_gradient = 0
    n = float(len(points))
    for i in range(len(points)):
        x = points.iloc[i].Age
        y = points.iloc[i].Premium
        m_gradient += -(2/n) * x * (y - (m*x + b))
        b_gradient += -(2/n) * (y - (m*x + b))
    new_m = m - (learning_rate * m_gradient)
    new_b = b - (learning_rate * b_gradient)
    return new_m, new_b

if __name__ == "__main__":
    data = pd.read_csv("./datasets/simplelinearregression.csv")
    m = 0  
    b = 0
    learning_rate = 0.00000001
    num_iterations = 500000
    for i in range(num_iterations):
        m, b = gradient_descent(m, b, data, learning_rate)
        if i % 100 == 0:
            print(f"Iteration {i}: m = {m}, b = {b}, loss = {loss_function(m, b, data)}")
    plt.scatter(data.Age, data.Premium, color='blue')
    plt.plot(data.Age, m*data.Age + b, color='red')
    plt.xlabel("Age")
    plt.ylabel("Premium")
    plt.title("Linear Regression")
    plt.show()
