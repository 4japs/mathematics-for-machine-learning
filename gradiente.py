import numpy as np
import matplotlib.pyplot as plt

grau = int(input("Digite o grau do polinômio: "))

coeficientes = []
print("\nDigite os coeficientes do polinômio:")
for i in range(grau + 1):
    c = float(input(f"Coeficiente a_{i} (para x^{i}): "))
    coeficientes.append(c)

x_inicial = float(input("\nDigite o valor inicial de x (chute inicial): "))

lmbd = 0.06
gama = 0.07

f = np.polynomial.Polynomial(coeficientes)
df = f.deriv()

print(f"\nFunção montada: f(x) = {f}")
print(f"Derivada calculada: f'(x) = {df}")

historico_x = [x_inicial]
historico_y = [f(x_inicial)]

x_atual = x_inicial
v = 0.0

for _ in range(1000):
    gradiente = df(x_atual)

    if gradiente == 0:
        break

    v_prox = gama * v + lmbd * gradiente
    prox = x_atual - v_prox

    if f(x_atual) <= f(prox):
        lmbd /= 10
        v = 0.0
        continue

    x_atual = prox
    v = v_prox
    
    historico_x.append(x_atual)
    historico_y.append(f(x_atual))

margem = max(2.0, abs(max(historico_x) - min(historico_x)) * 0.5)
x_min = min(historico_x) - margem
x_max = max(historico_x) + margem

x_curva = np.linspace(x_min, x_max, 500)
y_curva = f(x_curva)
plt.figure(figsize=(10, 6))
plt.plot(x_curva, y_curva, label='Função f(x)', color='#1f77b4', linewidth=2)
plt.scatter(historico_x, historico_y, color='red', zorder=5, label='Trajetória do Gradiente')

for i in range(len(historico_x) - 1):
    plt.annotate('', 
                xy=(historico_x[i+1], historico_y[i+1]), 
                xytext=(historico_x[i], historico_y[i]),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

plt.title(f'Gradiente Descendente com Momento (Polinômio de Grau {grau})')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

print(f"\nResultado final: O algoritmo parou em x = {x_atual:.6f} com f(x) = {f(x_atual):.6f}")

plt.show()