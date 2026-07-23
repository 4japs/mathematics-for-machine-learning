import numpy as np
import matplotlib.pyplot as plt
import sys

def limpar_tela():
    print("\n" * 2)

def obter_matriz_usuario():
    print("\n--- Definir Matriz 2x2 ---")

    entrada = input("Entrada (formato a,b; c,d): ").strip()
    try:
        matriz = np.matrix(entrada)
        if matriz.shape != (2, 2):
            print("Erro: Insira uma matriz 2x2.")
            return None
        return np.array(matriz)
    except Exception as e:
        print(f"Erro: {e}")
        return None

def gerar_circulo(pontos=50):
    
    theta = np.linspace(0, 2*np.pi, pontos)
    x = np.cos(theta)
    y = np.sin(theta)
    return np.vstack([x, y])

def plotar_transformacao(matriz):

    
    x = np.linspace(0, 1, 10); y = np.linspace(0, 1, 10)
    X, Y = np.meshgrid(x, y)
    pontos = np.vstack([X.flatten(), Y.flatten()])
    
    pontos_trans = matriz @ pontos
    
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(pontos[0, :], pontos[1, :], c='blue', alpha=0.5)
    plt.title("Original")
    plt.grid(True); plt.axis('equal')

    plt.subplot(1, 2, 2)
    plt.scatter(pontos_trans[0, :], pontos_trans[1, :], c='red', alpha=0.5)
    plt.title("Transformado")
    plt.grid(True); plt.axis('equal')
    plt.show()

def visualizar_svd(matriz):
    U, S, Vt = np.linalg.svd(matriz)
    Sigma = np.diag(S)
    
    x = np.linspace(0, 1, 10); y = np.linspace(0, 1, 10)
    X, Y = np.meshgrid(x, y)
    pontos = np.vstack([X.flatten(), Y.flatten()])
    
    passo1 = Vt @ pontos
    passo2 = Sigma @ passo1
    passo3 = U @ passo2 
    
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    estados = [
        (pontos, "Original", 'blue'),
        (passo1, "Rotação (Vt)", 'green'),
        (passo2, "Escala (Sigma)", 'orange'),
        (passo3, "Rotação (U)", 'red')
    ]
    
    for i, (dados, titulo, cor) in enumerate(estados):
        axs[i].scatter(dados[0, :], dados[1, :], c=cor, alpha=0.6)
        axs[i].axhline(0, color='k', lw=0.5); axs[i].axvline(0, color='k', lw=0.5)
        axs[i].set_title(titulo)
        axs[i].axis('equal'); axs[i].grid(True, linestyle='--')
        
    plt.tight_layout()
    plt.show()

def visualizar_convergencia(matriz):

    vals, vecs = np.linalg.eig(matriz)
    
    idx_dom = np.argmax(np.abs(vals))
    autovetor_dom = vecs[:, idx_dom]
    autovalor_dom = vals[idx_dom]
    
    if np.iscomplex(autovalor_dom):
        print("Aviso: Autovalores complexos detectados.")

    pontos = gerar_circulo(pontos=40)
    historico = [pontos]
    
    iteracoes = 10
    pontos_atuais = pontos.copy()
    
    for _ in range(iteracoes):
        novos_pontos = matriz @ pontos_atuais
        
        normas = np.linalg.norm(novos_pontos, axis=0)
        normas[normas == 0] = 1 
        novos_pontos = novos_pontos / normas
        
        pontos_atuais = novos_pontos
        historico.append(pontos_atuais)

    plt.figure(figsize=(8, 8))
    
    for i, pts in enumerate(historico):
        alpha = (i + 1) / len(historico)
        cor = plt.cm.Reds(alpha)
        label = "Inicial" if i == 0 else ("Final" if i == iteracoes else None)
        plt.scatter(pts[0, :], pts[1, :], color=cor, s=20, label=label)

    x_v = [autovetor_dom[0] * -2, autovetor_dom[0] * 2]
    y_v = [autovetor_dom[1] * -2, autovetor_dom[1] * 2]
    plt.plot(x_v, y_v, color='green', linestyle='--', linewidth=2, label=f'Autovetor Dominante ($\lambda={autovalor_dom:.2f}$)')

    plt.axhline(0, color='black', lw=0.5)
    plt.axvline(0, color='black', lw=0.5)
    plt.title(f"Convergência após {iteracoes} iterações (Normalizado)")
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    matriz_atual = None
    
    while True:
        print("\n=== VISUALIZADOR DE ÁLGEBRA LINEAR ===")
        status = "Indefinida" if matriz_atual is None else "\n" + str(matriz_atual)
        print(f"Matriz Atual: {status}")
        print("-" * 30)
        print("1. Definir Nova Matriz")
        print("2. Visualizar Transformação Básica")
        print("3. Visualizar Decomposição SVD")
        print("4. Visualizar Convergência")
        print("0. Sair")
        
        opcao = input("Escolha: ")
        
        if opcao == '1':
            nova = obter_matriz_usuario()
            if nova is not None: matriz_atual = nova
        elif opcao == '2':
            if matriz_atual is not None: plotar_transformacao(matriz_atual)
            else: print("Defina a matriz primeiro.")
        elif opcao == '3':
            if matriz_atual is not None: visualizar_svd(matriz_atual)
            else: print("Defina a matriz primeiro.")
        elif opcao == '4':
            if matriz_atual is not None: visualizar_convergencia(matriz_atual)
            else: print("Defina a matriz primeiro.")
        elif opcao == '0':
            sys.exit()
        else:
            print("Opção inválida.")

if __name__ == "__main__":
    main()