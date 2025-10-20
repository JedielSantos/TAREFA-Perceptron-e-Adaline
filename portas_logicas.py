import numpy as np

# ----------------------------
# 1. Dados das portas lógicas
# ----------------------------
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

Y_and = np.array([0, 0, 0, 1])
Y_or  = np.array([0, 1, 1, 1])
Y_xor = np.array([0, 1, 1, 0])

# Adiciona coluna de bias (1)
Xb = np.hstack((np.ones((X.shape[0], 1)), X))

# ----------------------------
# 2. Função de ativação (degrau)
# ----------------------------
def step_function(x):
    return np.where(x >= 0, 1, 0)

# ----------------------------
# 3. Perceptron
# ----------------------------
def perceptron_train(X, y, eta=0.1, epochs=50):
    w = np.random.uniform(-0.5, 0.5, X.shape[1])
    for epoch in range(epochs):
        total_error = 0
        for xi, target in zip(X, y):
            y_pred = step_function(np.dot(w, xi))
            erro = target - y_pred
            w += eta * erro * xi
            total_error += erro ** 2
        if total_error == 0:
            break
    return w, epoch + 1, total_error

# ----------------------------
# 4. Adaline
# ----------------------------
def adaline_train(X, y, eta=0.01, epochs=1000):
    w = np.random.uniform(-0.5, 0.5, X.shape[1])
    for epoch in range(epochs):
        y_pred = np.dot(X, w)
        erro = y - y_pred
        mse = np.mean(erro ** 2)
        w += eta * np.dot(X.T, erro)
        if mse < 0.0001:
            break
    return w, epoch + 1, mse

# ----------------------------
# 5. Executar e mostrar resultados
# ----------------------------
def run_tests(name, y):
    print(f"\n===== PORTA {name} =====")
    
    w_p, ep_p, err_p = perceptron_train(Xb, y)
    print(">> Perceptron")
    print(f"Pesos finais: {w_p}")
    print(f"Épocas: {ep_p}")
    print(f"Erro final: {err_p}")

    w_a, ep_a, err_a = adaline_train(Xb, y)
    print("\n>> Adaline")
    print(f"Pesos finais: {w_a}")
    print(f"Épocas: {ep_a}")
    print(f"Erro final (MSE): {err_a:.6f}")

# ----------------------------
# 6. Rodar testes
# ----------------------------
run_tests("AND", Y_and)
run_tests("OR", Y_or)
run_tests("XOR", Y_xor)