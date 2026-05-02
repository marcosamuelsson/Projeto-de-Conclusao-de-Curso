import numpy as np
import os
from PIL import Image
from tkinter import filedialog, messagebox

class CNNDeepLearning:
    def __init__(self, input_size=(64, 64)):
        self.input_size = input_size
        # Filtro fixo de detecção de bordas (ajuda já que não estamos treinando o kernel via backprop complexo)
        self.kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]) * 0.1
        
        self.flat_size = ((input_size[0] - 2) // 2) * ((input_size[1] - 2) // 2)
        self.w_densa = np.random.randn(self.flat_size, 1) * 0.01
        self.bias = 0

    def relu(self, x): return np.maximum(0, x)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def convolucao(self, img):
        h, w = img.shape
        saida = np.zeros((h - 2, w - 2))
        for i in range(h - 2):
            for j in range(w - 2):
                saida[i, j] = np.sum(img[i:i+3, j:j+3] * self.kernel)
        return saida

    def max_pooling(self, img):
        h, w = img.shape
        saida = np.zeros((h // 2, w // 2))
        for i in range(0, h, 2):
            for j in range(0, w, 2):
                if i+1 < h and j+1 < w:
                    saida[i//2, j//2] = np.max(img[i:i+2, j:j+2])
        return saida

    def treinar_imagem(self, img_path, label, lr=0.01):
        try:
            img = Image.open(img_path).convert('L').resize(self.input_size)
            img_array = np.array(img) / 255.0
            
            # Forward
            c1 = self.convolucao(img_array)
            a1 = self.relu(c1)
            p1 = self.max_pooling(a1)
            flattened = p1.flatten().reshape(-1, 1)
            
            z = np.dot(flattened.T, self.w_densa) + self.bias
            pred = self.sigmoid(z)
            
            # Backpropagation (Simplificado para a camada densa)
            erro = pred - label
            self.w_densa -= lr * flattened * erro
            self.bias -= lr * erro
            return float(np.abs(erro))
        except Exception as e:
            return None

    def salvar_modelo(self):
        np.save('modelo_pesos.npy', self.w_densa)
        np.save('modelo_bias.npy', self.bias)
        print("\n[INFO] Pesos salvos com sucesso!")

    def carregar_modelo(self):
        if os.path.exists('modelo_pesos.npy'):
            self.w_densa = np.load('modelo_pesos.npy')
            self.bias = np.load('modelo_bias.npy')
            print("[INFO] Modelo carregado!")

    def predizer(self, img_path):
        img = Image.open(img_path).convert('L').resize(self.input_size)
        img_array = np.array(img) / 255.0
        c1 = self.convolucao(img_array)
        p1 = self.max_pooling(self.relu(c1))
        f = p1.flatten().reshape(-1, 1)
        prob = self.sigmoid(np.dot(f.T, self.w_densa) + self.bias)
        return prob[0][0]

# --- FLUXO DE EXECUÇÃO ---

cnn = CNNDeepLearning()

# print("--- FASE DE TREINAMENTO ---")
# print("Selecione a pasta que contém apenas fotos de GATOS.")
# dir_cats = filedialog.askdirectory(title="Pasta de GATOS", initialdir="/home/marco-samuelsson/Documentos/BCC/Projeto-de-Conclusao-de-Curso")
# print("Selecione a pasta que contém fotos de OUTRAS COISAS (Cachorros, etc).")
# dir_others = filedialog.askdirectory(title="Pasta de NÃO-GATOS", initialdir="/home/marco-samuelsson/Documentos/BCC/Projeto-de-Conclusao-de-Curso")

# if dir_cats and dir_others:
#     epocas = 5  # Quantas vezes o modelo verá todo o conjunto de dados
#     for epoca in range(epocas):
#         erro_total = 0
#         cont = 0
#         print(f"\nIniciando Época {epoca+1}/{epocas}...")
        
#         # Treinar Gatos (Label 1)
#         for arq in os.listdir(dir_cats):
#             res = cnn.treinar_imagem(os.path.join(dir_cats, arq), label=1)
#             if res is not None: 
#                 erro_total += res
#                 cont += 1
        
#         # Treinar Não-Gatos (Label 0)
#         for arq in os.listdir(dir_others):
#             res = cnn.treinar_imagem(os.path.join(dir_others, arq), label=0)
#             if res is not None: 
#                 erro_total += res
#                 cont += 1
        
#         print(f"Erro médio da época: {erro_total/cont:.4f}")

#     cnn.salvar_modelo()
#     messagebox.showinfo("Sucesso", "Treinamento concluído!")

# --- FASE DE TESTE ---
print("\n--- FASE DE TESTE ---")
teste_path = filedialog.askopenfilename(title="Selecione uma imagem para testar", 
                                        initialdir="/home/marco-samuelsson/Documentos/BCC/Projeto-de-Conclusao-de-Curso")

if teste_path:
    cnn.carregar_modelo()
    resultado = cnn.predizer(teste_path)
    print(f"\nCaminho: {teste_path}")
    print(f"Confiança: {resultado * 100:.2f}%")
    print("Veredito:", "GATO" if resultado > 0.5 else "NÃO É GATO")