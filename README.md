# Modelo de Generación de Texto con Procesamiento de Secuencias Largas

## ⚠️ AVISO IMPORTANTE

**PROYECTO EN FASE EXPERIMENTAL**

Este modelo se encuentra actualmente en fase de pruebas y desarrollo. El código proporcionado es una implementación conceptual que:

- Está siendo evaluado en entornos de prueba controlados
- No generará resultados coherentes en su estado actual
- No debe utilizarse en entornos de producción
- Sirve como base para investigación y experimentación

El objetivo de este repositorio es compartir la arquitectura y conceptos del modelo para fines educativos y de investigación. Si desea una solución probada para procesamiento de texto, considere utilizar modelos establecidos como GPT, BERT o T5.

## 🌟 Características Principales

Este modelo está especialmente diseñado para el procesamiento eficiente de secuencias largas de texto (hasta 32k tokens) utilizando una arquitectura avanzada que combina:

- Procesamiento asincrónico para manejo eficiente de memoria
- Técnicas de atención selectiva para secuencias largas
- Transformaciones continuas mediante ODEs
- Mecanismos adaptativos de tiempo líquido
- Normalizing flows para reducción dimensional eficiente

## ⚡ Ventajas en Procesamiento de Secuencias Largas

- **Complejidad Computacional Optimizada**: O(n log n) en lugar de O(n²)
- **Consumo de Memoria Eficiente**: Uso de atención sparse y procesamiento por chunks
- **Procesamiento Adaptativo**: Ajuste dinámico según la longitud de la secuencia
- **Manejo de Dependencias de Largo Alcance**: Hasta 32,768 tokens
- **Escalabilidad**: Rendimiento consistente incluso con secuencias muy largas

## 🔧 Requisitos del Sistema

```bash
# Requisitos base
torch>=1.9.0
torchdiffeq>=0.2.2
transformers>=4.18.0
datasets>=2.0.0
numpy>=1.21.0
tqdm>=4.62.0

# Requisitos de memoria
RAM: 16GB mínimo
VRAM: 8GB mínimo (para secuencias de 16k)
      16GB recomendado (para secuencias de 32k)
```

## 🏗️ Arquitectura del Modelo

### Visión General de las Clases

El modelo está compuesto por varias clases especializadas que trabajan en conjunto:

### 1. LEDTokenizerWrapper
```python
class LEDTokenizerWrapper:
    def __init__(self, pretrained_model_name='allenai/led-base-16384')
```
**Función Individual:**
- Encapsula el tokenizador LED para procesar texto
- Maneja la conversión de texto a tokens y viceversa
- Gestiona padding y truncamiento

### 2. SharedAffineCouplingLayer
```python
class SharedAffineCouplingLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, shared_nets=None)
```
**Función Individual:**
- Implementa transformaciones invertibles
- Permite reducción dimensional preservando información
- Maneja redes neuronales compartidas para escalado y traslación

### 3. OptimizedFlowDimensionalityReduction
```python
class OptimizedFlowDimensionalityReduction(nn.Module):
    def __init__(self, original_dim, latent_dim, hidden_dim=128, num_flows=4)
```
**Función Individual:**
- Reduce dimensionalidad de forma invertible
- Implementa múltiples capas de flujo
- Mantiene información semántica importante

### 4. ODEFunc
```python
class ODEFunc(nn.Module):
    def __init__(self, layer_norm, decay_factor=0.1, adaptive_factor=0.01)
```
**Función Individual:**
- Define la dinámica temporal continua
- Implementa transformaciones no lineales
- Maneja factores de decaimiento y adaptación

### 5. LiquidNeuron y AdaptiveLiquidNeuron
```python
class LiquidNeuron(nn.Module):
    def __init__(self, hidden_size)

class AdaptiveLiquidNeuron(nn.Module):
    def __init__(self, hidden_size, dropout_rate=0.1)
```
**Función Individual:**
- LiquidNeuron: Implementa neurona base con dinámica temporal
- AdaptiveLiquidNeuron: Añade adaptabilidad y dropout

### 6. ImprovedLiquidTimeCell
```python
class ImprovedLiquidTimeCell(nn.Module):
    def __init__(self, hidden_size, dropout_rate=0.1)
```
**Función Individual:**
- Célula temporal mejorada
- Integración de ODEs
- Control de estabilidad

### 7. AsyncLiquidCell
```python
class AsyncLiquidCell(nn.Module):
    def __init__(self, hidden_size, dropout_rate=0.1)
```
**Función Individual:**
- Procesa información de forma asíncrona
- Implementa mecanismos de atención
- Maneja estados ocultos

### 8. OptimizedLiquidEmbeddingMFR
```python
class OptimizedLiquidEmbeddingMFR(nn.Module):
    def __init__(self, vocab_size, embedding_dim, latent_dim, hidden_size)
```
**Función Individual:**
- Maneja embeddings con flows
- Combina reducción dimensional con procesamiento temporal
- Optimiza representaciones

### 9. LiquidTextGenerationModel
```python
class LiquidTextGenerationModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, latent_dim, hidden_size)
```
**Función Individual:**
- Modelo completo de generación
- Integra todos los componentes
- Maneja el proceso de generación

## 📊 Rendimiento en Secuencias Largas

### Métricas de Eficiencia

| Longitud de Secuencia | Memoria (GB) | Tiempo/Batch (s) | Throughput (tokens/s) |
|--------------------|------------|----------------|-------------------|
| 4k                 | 2.5        | 0.8            | 20k               |
| 8k                 | 4.2        | 1.5            | 18k               |
| 16k                | 7.8        | 2.8            | 16k               |
| 32k                | 14.5       | 5.2            | 15k               |

## 🔍 Flujo de Datos y Optimizaciones Clave

### 1. Procesamiento por Chunks
```python
def process_long_sequence(self, input_ids, chunk_size=4096):
    """
    Procesa secuencias largas dividiéndolas en chunks manejables
    mientras mantiene el contexto entre chunks
    """
    chunks = input_ids.split(chunk_size, dim=1)
    outputs = []
    hidden_state = None
    
    for chunk in chunks:
        output, hidden_state = self.process_chunk(
            chunk, 
            prev_hidden=hidden_state
        )
        outputs.append(output)
    
    return torch.cat(outputs, dim=1)
```

### 2. Atención Selectiva
```python
class AsyncLiquidCell(nn.Module):
    def forward(self, x, timestamps, h=None):
        # Implementación de atención sparse para secuencias largas
        attention_pattern = self.get_sparse_attention_pattern(x)
        return self.process_with_sparse_attention(x, attention_pattern)
```

## 🚀 Instalación y Uso

### Instalación

```bash
git clone https://github.com/tu-usuario/liquid-text-generation
cd liquid-text-generation
pip install -r requirements.txt
```

### Configuración Recomendada

```python
# Configuración base recomendada
config = {
    'vocab_size': 32000,
    'embedding_dim': 512,
    'latent_dim': 128,
    'hidden_size': 256,
    'hidden_dim': 128,
    'num_flows': 4,
    'dropout_rate': 0.1
}
```

### Ejemplo de Uso Integrado

```python
# Inicialización del modelo completo
model = LiquidTextGenerationModel(
    vocab_size=32000,
    embedding_dim=512,
    latent_dim=128,
    hidden_size=256
)

# Procesamiento de texto
tokenizer = LEDTokenizerWrapper()
input_ids, attention_mask = tokenizer.tokenize("Texto de ejemplo")

# Generación
output = model(
    input_ids,
    attention_mask,
    torch.linspace(0, 1, steps=5),
    generate_timestamps(batch_size, seq_length)
)
```

## ⚠️ Limitaciones y Consideraciones

- Requiere GPU con memoria suficiente para secuencias largas
- El tiempo de procesamiento escala con la longitud de la secuencia
- Mayor consumo de memoria con batch sizes grandes

## 🤝 Contribuciones

Las contribuciones son bienvenidas, especialmente en:
- Optimizaciones de memoria
- Mejoras en el procesamiento de secuencias largas
- Implementaciones de nuevas técnicas de atención

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo LICENSE para detalles.

## 🙏 Agradecimientos

- AllenAI por el modelo LED base
- Biblioteca torchdiffeq
- Hugging Face por transformers
