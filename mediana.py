import numpy as np
import matplotlib.pyplot as plt
import math


# Primeira Gaussiana
w1, media1, sigma1 = 0.2, 10.0, np.sqrt(1.0)

# Segunda Gaussiana
w2, media2, sigma2 = 0.8, 0.0, np.sqrt(8.4)

erf_vec = np.vectorize(math.erf)

def pdf_mistura(x):
    pdf1 = (1.0 / (sigma1 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - media1) / sigma1)**2)
    pdf2 = (1.0 / (sigma2 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - media2) / sigma2)**2)
    return w1 * pdf1 + w2 * pdf2

def cdf_mistura(x):
    cdf1 = 0.5 * (1 + erf_vec((x - media1) / (sigma1 * np.sqrt(2))))
    cdf2 = 0.5 * (1 + erf_vec((x - media2) / (sigma2 * np.sqrt(2))))
    return w1 * cdf1 + w2 * cdf2


l = -10.0
r = 20.0
eps = 1e-5

chutes_x = []
chutes_y = []

while (r - l) > eps:

    ponto_medio = (l + r) / 2.0
    valor_cdf = cdf_mistura(ponto_medio)
    
    chutes_x.append(ponto_medio)
    chutes_y.append(valor_cdf)
    
    if valor_cdf < 0.5:
        l = ponto_medio
    else:
        r = ponto_medio

mediana_final = chutes_x[-1]
print(f"Mediana encontrada: {mediana_final:.4f} em {len(chutes_x)} iterações.")


x_vals = np.linspace(-10, 20, 10000)
pdf_vals = pdf_mistura(x_vals)
cdf_vals = cdf_mistura(x_vals)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

ax1.plot(x_vals, pdf_vals, color='black', linewidth=2, label='PDF da Mistura')
ax1.fill_between(x_vals, pdf_vals, where=(x_vals <= mediana_final), color='gray', alpha=0.3, label='50% da Área')
ax1.axvline(mediana_final, color='red', linestyle='--', linewidth=2, label=f'Mediana ({mediana_final:.2f})')
ax1.set_title('Densidade de Probabilidade (PDF) - Visão 2D')
ax1.set_xlabel('x')
ax1.set_ylabel('Densidade')
ax1.legend()
ax1.grid(alpha=0.3)

ax2.plot(x_vals, cdf_vals, color='black', linewidth=2, label='CDF da Mistura', zorder=1)
ax2.axhline(0.5, color='gray', linestyle='--', linewidth=1.5, label='Alvo (0.5)')

cores = plt.cm.coolwarm(np.linspace(0, 1, len(chutes_x)))

# Plotando os chutes iterativamente
for i in range(len(chutes_x)):
    x_c = chutes_x[i]
    y_c = chutes_y[i]

    ax2.scatter(x_c, y_c, color=cores[i], s=50, zorder=3, edgecolors='black')
    
    ax2.vlines(x_c, ymin=0, ymax=y_c, color=cores[i], linestyle=':', alpha=0.7, zorder=2)
    
    if i < 6:
        ax2.annotate(f'{i+1}', (x_c, y_c), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9, fontweight='bold', color=cores[i])

ax2.set_title('Busca Binária na Função Acumulada (CDF)')
ax2.set_xlabel('x (Chutes do Algoritmo)')
ax2.set_ylabel('Probabilidade Acumulada')
ax2.set_ylim(0, 1.05)
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.show()