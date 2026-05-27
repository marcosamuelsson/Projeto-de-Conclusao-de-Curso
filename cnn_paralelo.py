import os
# Silencia os avisos iniciais do TensorFlow e do oneDNN
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- 1. CARREGAMENTO E PREPARAÇÃO DOS DADOS ---
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

# Criação de geradores de lote (Batches) do TensorFlow para o loop customizado
BATCH_SIZE = 64
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(10000).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(BATCH_SIZE)

# --- 2. DEFINIÇÃO DA ARQUITETURA DA CNN ---
def build_cnn():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10)
    ])
    return model

# --- 3. ALGORITMO CLIP-Q (PODA E QUANTIZAÇÃO PARALELA) ---
def quantize_chunk(chunk, bits):
    """Aplica a quantização uniforme linear nos pesos sobreviventes."""
    if chunk.size == 0: 
        return chunk
    min_val, max_val = np.min(chunk), np.max(chunk)
    if min_val == max_val: 
        return chunk
    
    levels = 2**bits - 1
    normalized = (chunk - min_val) / (max_val - min_val)
    quantized = np.round(normalized * levels) / levels
    return quantized * (max_val - min_val) + min_val

def apply_clip_q_inline(model, bits=4, pruning_ratio=0.3, num_threads=4):
    """
    Varre as camadas do modelo, aplica a poda e dispara múltiplas 
    threads para calcular a quantização em paralelo.
    """
    for layer in model.layers:
        if isinstance(layer, (layers.Conv2D, layers.Dense)):
            weights_list = layer.get_weights()
            if not weights_list or len(weights_list) == 0: 
                continue
            
            W, b = weights_list[0], weights_list[1]
            
            # 1. Poda (Pruning) baseado no percentil de magnitude
            abs_W = np.abs(W)
            threshold = np.percentile(abs_W, pruning_ratio * 100)
            mask_survived = abs_W >= threshold
            surviving_weights = W[mask_survived]
            
            if surviving_weights.size == 0: 
                continue
                
            # 2. Quantização Paralela (Threads) dos pesos sobreviventes
            chunks = np.array_split(surviving_weights, num_threads)
            quantized_chunks = [None] * num_threads
            
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = {executor.submit(quantize_chunk, chunks[i], bits): i for i in range(num_threads)}
                for future in as_completed(futures):
                    idx = futures[future]
                    quantized_chunks[idx] = future.result()
            
            # Recombina os pesos e reconstrói o tensor original (zerando os podados)
            new_W = np.zeros_like(W)
            new_W[mask_survived] = np.concatenate(quantized_chunks)
            
            # Atualiza os pesos da camada diretamente no modelo em treinamento
            layer.set_weights([new_W, b])

# --- 4. FUNÇÃO DE AVALIAÇÃO DE ACURÁCIA ---
def evaluate_accuracy(model, dataset):
    acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    for x_batch, y_batch in dataset:
        logits = model(x_batch, training=False)
        acc_metric.update_state(y_batch, logits)
    return acc_metric.result().numpy()

# --- 5. PIPELINE DE TREINAMENTO CUSTOMIZADO ---
def train_model(model, optimizer, loss_fn, epochs=10, optimize_clip_q=False, num_threads=4):
    print(f"\nIniciando Treinamento {'COM CLIP-Q Ativo' if optimize_clip_q else 'PADRÃO (Original)'}...")
    
    acc_history = []
    for epoch in range(epochs):
        start_time = time.time()
        
        # Métricas de treino para a época
        epoch_loss = tf.keras.metrics.Mean()
        epoch_acc = tf.keras.metrics.SparseCategoricalAccuracy()
        
        # Loop sobre os batches
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                logits = model(x_batch_train, training=True)
                loss_value = loss_fn(y_batch_train, logits)
                
            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            
            epoch_loss.update_state(loss_value)
            epoch_acc.update_state(y_batch_train, logits)
        
        # INTERVENÇÃO DO CLIP-Q: Aplica a poda e quantização logo após o ajuste de gradientes da época
        if optimize_clip_q:
            apply_clip_q_inline(model, bits=4, pruning_ratio=0.30, num_threads=num_threads)
            
        # Avalia a acurácia de validação (teste) ao fim da época
        val_accuracy = evaluate_accuracy(model, test_dataset)
        acc_history.append(val_accuracy)
        
        print(f"Época {epoch+1}/{epochs} - Tempo: {time.time() - start_time:.2f}s - Loss: {epoch_loss.result():.4f} - Train Acc: {epoch_acc.result():.4f} - Val Acc: {val_accuracy:.4f}")
        
    return acc_history

# --- 6. EXECUÇÃO COMPARATIVA ---

# Configurações Comuns
loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
EPOCHS = 10
NUM_THREADS = 4  # Quantidade de threads alocadas para a quantização

# Cenário A: Modelo Original (Sem nenhuma alteração nos pesos durante o treino)
model_original = build_cnn()
optimizer_orig = tf.keras.optimizers.Adam()
history_orig = train_model(model_original, optimizer_orig, loss_function, epochs=EPOCHS, optimize_clip_q=False)

# Cenário B: Modelo Otimizado com CLIP-Q Inline (Poda e Quantização em tempo de treino)
model_clipq = build_cnn()
optimizer_clipq = tf.keras.optimizers.Adam()
history_clipq = train_model(model_clipq, optimizer_clipq, loss_function, epochs=EPOCHS, optimize_clip_q=True, num_threads=NUM_THREADS)

# --- 7. RELATÓRIO FINAL E VISUALIZAÇÃO ---
print("\n" + "="*55)
print("               RELATÓRIO DE RESULTADOS FINAL             ")
print("="*55)
print(f"Número de Threads utilizadas na Otimização: {NUM_THREADS}")
print(f"Acurácia Final do Modelo Original:            {history_orig[-1]:.4f}")
print(f"Acurácia Final do Modelo CLIP-Q (Paralelo):   {history_clipq[-1]:.4f}")
print(f"Diferença de Impacto na Acurácia:             {history_clipq[-1] - history_orig[-1]:+.4f}")
print("="*55)

# Plotagem Gráfica das Épocas
plt.figure(figsize=(10, 5))
plt.plot(range(1, EPOCHS+1), history_orig, marker='o', label='Acurácia Original (Val)')
plt.plot(range(1, EPOCHS+1), history_clipq, marker='s', label='Acurácia CLIP-Q Inline (Val)')
plt.xlabel('Época')
plt.ylabel('Acurácia')
plt.title('Comparativo de Treinamento: Original vs. Otimização CLIP-Q em Tempo de Execução')
plt.legend()
plt.grid(True)
plt.show()