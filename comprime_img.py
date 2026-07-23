import numpy as np
import matplotlib.pyplot as plt
from skimage import data, color

img = color.rgb2gray(data.astronaut())
m, n = img.shape
print(f"Dimensões originais: {m}x{n} (Total de pixels: {m*n})")

U, S, Vt = np.linalg.svd(img, full_matrices=False)

S_diag = np.diag(S)

k_valores = [20, 75, 200]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

axes[0].imshow(img, cmap='gray')
axes[0].set_title(f'Original (Posto {min(m, n)})')
axes[0].axis('off')

for idx, k in enumerate(k_valores):
    
    img_reconstruida = U[:, :k] @ S_diag[:k, :k] @ Vt[:k, :]
    
    # Cálculo da Taxa de Compressão
    valores_armazenados = (m * k) + k + (k * n)
    compressao = (1 - (valores_armazenados / (m * n))) * 100
    print(f"Para k={k}: Foram retidos {valores_armazenados} valores. Compressão de {compressao:.1f}%")
    
    # Plota a aproximação
    ax = axes[idx + 1]
    ax.imshow(img_reconstruida, cmap='gray')
    ax.set_title(f'Reconstruída (k={k})\nCompressão: {compressao:.1f}%')
    ax.axis('off')

plt.suptitle('Comparação do Truncamento SVD (Aproximação de Posto k)', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96], h_pad=3.0)
plt.show()