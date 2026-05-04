import numpy as np
import os
import json
from PIL import Image
from tkinter import filedialog, messagebox

class CNNDeepLearning:
    def __init__(self, input_size=(64, 64)):
        self.input_size = input_size
        # Kernel de detecção de bordas ligeiramente mais forte para realçar silhuetas
        self.kernel = np.array([
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ])
        
        # Cálculo do tamanho do vetor após Convolução (64-2=62) e Pooling (62/2=31)
        # 31 * 31 = 961 entradas na camada densa
        self.flat_size = ((input_size[0] - 2) // 2) * ((input_size[1] - 2) // 2)
        
        # Inicialização de Xavier/Glorot simplificada para melhor convergência
        self.w_densa = np.random.randn(self.flat_size, 1) * np.sqrt(1 / self.flat_size)
        self.bias = 0.0

    def relu(self, x): 
        return np.maximum(0, x)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -20, 20)))

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

    def preparar_imagem(self, img_path):
        # Suporta qualquer tamanho e cor transformando em Escala de Cinza Padronizada
        img = Image.open(img_path).convert('L').resize(self.input_size)
        img_array = np.array(img) / 255.0
        # Normalização extra: subtrair a média ajuda na convergência
        return img_array - np.mean(img_array)

    def treinar_imagem(self, img_path, label, lr=0.05): # Aumentei um pouco o learning rate
        try:
            img_array = self.preparar_imagem(img_path)
            
            # Forward Pass
            c1 = self.convolucao(img_array)
            a1 = self.relu(c1)
            p1 = self.max_pooling(a1)
            flattened = p1.flatten().reshape(-1, 1)
            
            z = np.dot(flattened.T, self.w_densa) + self.bias
            pred = self.sigmoid(z)
            
            # Backpropagation simplificado
            erro = pred - label
            self.w_densa -= lr * flattened * erro
            self.bias -= lr * float(erro)
            
            return float(np.abs(erro))
        except Exception as e:
            return None

    def salvar_modelo(self, filepath):
        dados = {
            "input_size": self.input_size,
            "bias": float(self.bias),
            "w_densa": self.w_densa.tolist()
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(dados, f, indent=4)
        print(f"\n[INFO] Pesos salvos em JSON: {filepath}")

    def carregar_modelo(self, filepath):
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                dados = json.load(f)
            self.bias = float(dados["bias"])
            self.w_densa = np.array(dados["w_densa"])
            print(f"[INFO] Modelo JSON carregado!")

    def predizer(self, img_path):
        img_array = self.preparar_imagem(img_path)
        c1 = self.convolucao(img_array)
        a1 = self.relu(c1)
        p1 = self.max_pooling(a1)
        f = p1.flatten().reshape(-1, 1)
        z = np.dot(f.T, self.w_densa) + self.bias
        return self.sigmoid(z)[0][0]

# --- FLUXO DE EXECUÇÃO ---

if __name__ == "__main__":
    cnn = CNNDeepLearning()
    path_json = "/home/marco-samuelsson/Documentos/BCC/Projeto-de-Conclusao-de-Curso/modelo_pesos.json"

    print("--- FASE DE TREINAMENTO ---")
    dir_cats = filedialog.askdirectory(title="Pasta de GATOS", initialdir="/home/marco-samuelsson/Documentos/BCC/Projeto-de-Conclusao-de-Curso")
    dir_others = filedialog.askdirectory(title="Pasta de NÃO-GATOS", initialdir="/home/marco-samuelsson/Documentos/BCC/Projeto-de-Conclusao-de-Curso")

    extensoes_validas = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')

    if dir_cats and dir_others:
        epocas = 2 # Aumentado para 10 para melhor aprendizado
        for epoca in range(epocas):
            erro_total, cont = 0, 0
            
            # Unindo caminhos para iterar de forma equilibrada
            dataset = [(dir_cats, 1), (dir_others, 0)]
            for pasta, label in dataset:
                for arq in os.listdir(pasta):
                    if arq.lower().endswith(extensoes_validas):
                        caminho = os.path.join(pasta, arq)
                        if os.path.isfile(caminho):
                            res = cnn.treinar_imagem(caminho, label)
                            if res is not None:
                                erro_total += res
                                cont += 1
            
            if cont > 0:
                print(f"Época {epoca+1}/{epocas} - Erro médio: {erro_total/cont:.4f}")

        cnn.salvar_modelo(path_json)
        messagebox.showinfo("Sucesso", "Treino finalizado!")

    print("\n--- FASE DE TESTE ---")
    teste_path = filedialog.askopenfilename(title="Selecione uma imagem para testar", initialdir="/home/marco-samuelsson/Documentos/BCC/Projeto-de-Conclusao-de-Curso")
    if teste_path:
        cnn.carregar_modelo(path_json)
        prob = cnn.predizer(teste_path)
        print(f"Confiança: {prob * 100:.2f}%")
        print("Resultado:", "GATO" if prob > 0.5 else "NÃO É GATO")