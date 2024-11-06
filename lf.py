import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint as odeint
import torch.optim as optim
import torch.utils.data as data
from transformers import LEDTokenizer, LEDModel
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import random
import math

# -----------------------------
# 1. Tokenizador de LED
# -----------------------------
class LEDTokenizerWrapper:
    def __init__(self, pretrained_model_name='allenai/led-base-16384'):
        self.tokenizer = LEDTokenizer.from_pretrained(pretrained_model_name)
        self.vocab_size = self.tokenizer.vocab_size
        self.pad_token_id = self.tokenizer.pad_token_id
        self.unk_token_id = self.tokenizer.unk_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.bos_token_id = self.tokenizer.bos_token_id

    def tokenize(self, sentences, max_length=512, padding='max_length', truncation=True):
        encoding = self.tokenizer(
            sentences,
            padding=padding,  # Puede ser 'max_length' o 'longest'
            truncation=truncation,
            max_length=max_length,
            return_tensors='pt'
        )
        return encoding['input_ids'], encoding['attention_mask']

    def decode(self, token_ids, skip_special_tokens=True):
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

# ------------------------------------
# 2. Creación de DataLoaders para Generación de Texto
# ------------------------------------
def create_dataloaders(batch_size=16, tokenizer=None, max_length=512, max_examples=10000, seed=42):
    # Establecer la semilla para reproducibilidad
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Cargar un dataset adecuado para generación de texto, por ejemplo, WikiText-2
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
    
    # Seleccionar un subconjunto del dataset si se especifica
    if max_examples is not None:
        # Asegurarse de que max_examples no exceda el tamaño del dataset
        total_available = len(dataset['train'])
        selected_examples = min(max_examples, total_available)
        dataset['train'] = dataset['train'].select(range(selected_examples))
        print(f"Dataset reducido a {selected_examples} ejemplos.")
    
    # Dividir el dataset en entrenamiento y validación (90/10)
    train_size = int(0.9 * len(dataset['train']))
    val_size = len(dataset['train']) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset['train'], [train_size, val_size])
    
    # Definir una clase Dataset compatible con PyTorch para generación de texto
    class TextGenerationDataset(data.Dataset):
        def __init__(self, hf_dataset, tokenizer, max_length):
            self.dataset = hf_dataset
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            text = self.dataset[idx]['text']
            # Tokenizar el texto y crear entradas y etiquetas desplazadas
            input_ids, attention_mask = self.tokenizer.tokenize([text], max_length=self.max_length)
            labels = input_ids.clone()
            return input_ids.squeeze(0), attention_mask.squeeze(0), labels.squeeze(0)

    # Crear instancias de los Datasets
    train_data = TextGenerationDataset(train_dataset, tokenizer, max_length)
    val_data = TextGenerationDataset(val_dataset, tokenizer, max_length)

    # Definir una función de collate para manejar el padding dinámico si es necesario
    def collate_fn(batch):
        input_ids = [item[0] for item in batch]
        attention_masks = [item[1] for item in batch]
        labels = [item[2] for item in batch]
        
        input_ids = torch.stack(input_ids)
        attention_masks = torch.stack(attention_masks)
        labels = torch.stack(labels)
        return input_ids, attention_masks, labels

    # Crear DataLoaders
    train_loader = data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_loader = data.DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    return train_loader, val_loader, tokenizer.vocab_size

# ------------------------------------
# 3. Definición de las Clases del Modelo para Generación de Texto
# ------------------------------------

# 3.1. SharedAffineCouplingLayer
class SharedAffineCouplingLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, shared_nets=None):
        super(SharedAffineCouplingLayer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        if shared_nets is None:
            self.scale_net = nn.Sequential(
                nn.Linear(input_dim // 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim // 2),
                nn.Tanh()
            )
            self.translate_net = nn.Sequential(
                nn.Linear(input_dim // 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim // 2)
            )
        else:
            self.scale_net = shared_nets['scale']
            self.translate_net = shared_nets['translate']

        self.scale_adapter = nn.Parameter(torch.ones(1))
        self.translate_adapter = nn.Parameter(torch.zeros(1))
        self.gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        B, S, D = x.size()
        x_reshaped = x.view(-1, D)
        x1, x2 = x_reshaped.chunk(2, dim=-1)

        s = self.scale_net(x1) * self.scale_adapter.view(1, 1)
        t = self.translate_net(x1) * self.translate_adapter.view(1, 1)

        s = s.view(B, S, -1)
        t = t.view(B, S, -1)
        x2 = x2.view(B, S, -1)

        y2 = x2 * torch.exp(s) + t
        y2 = y2 * self.gate + x2 * (1 - self.gate)

        y = torch.cat([x1.view(B, S, -1), y2], dim=-1)
        log_det_jacobian = torch.sum(s * self.gate, dim=-1)

        return y, log_det_jacobian

    def inverse(self, y):
        B, S, D = y.size()
        y_reshaped = y.view(-1, D)

        y1, y2 = y_reshaped.chunk(2, dim=-1)
        s = self.scale_net(y1) * self.scale_adapter
        t = self.translate_net(y1) * self.translate_adapter

        s = s.view(B, S, -1)
        t = t.view(B, S, -1)
        y2 = y2.view(B, S, -1)

        x2 = (y2 - t) * torch.exp(-s)
        x = torch.cat([y1.view(B, S, -1), x2], dim=-1)

        return x

# 3.2. OptimizedFlowDimensionalityReduction
class OptimizedFlowDimensionalityReduction(nn.Module):
    def __init__(self, original_dim, latent_dim, hidden_dim=128, num_flows=4, share_params=True):
        super(OptimizedFlowDimensionalityReduction, self).__init__()
        assert latent_dim < original_dim

        self.original_dim = original_dim
        self.latent_dim = latent_dim
        self.num_flows = num_flows

        self.proj_gate = nn.Parameter(torch.tensor(0.0))
        self.residual_scale = 0.1

        if share_params:
            shared_nets = {
                'scale': nn.Sequential(
                    nn.Linear(original_dim // 2, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, original_dim // 2),
                    nn.Tanh()
                ),
                'translate': nn.Sequential(
                    nn.Linear(original_dim // 2, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, original_dim // 2)
                )
            }
        else:
            shared_nets = None

        self.flows = nn.ModuleList([
            SharedAffineCouplingLayer(original_dim, hidden_dim, shared_nets)
            for _ in range(num_flows)
        ])

        self.initial_proj = nn.Linear(original_dim, original_dim)

    def forward(self, x):
        B, S, D = x.size()

        x_proj = self.initial_proj(x)
        x = x_proj * self.proj_gate + x * (1 - self.proj_gate)

        log_det_jacobian = torch.zeros(B, S).to(x.device)
        x_original = x

        for i, flow in enumerate(self.flows):
            y, ldj = flow(x)
            if i % 2 == 1:
                y = y + x_original * self.residual_scale
            x = y
            log_det_jacobian += ldj

        z = x[:, :, :self.latent_dim]
        if self.latent_dim == self.original_dim // 2:
            z = z + x_original[:, :, :self.latent_dim] * self.residual_scale

        return z, log_det_jacobian

    def inverse(self, z):
        D = self.original_dim
        z_full = torch.cat([z, torch.zeros_like(z[:, :, :D - self.latent_dim])], dim=-1)
        z_original = z_full

        for i, flow in reversed(list(enumerate(self.flows))):
            if i % 2 == 1:
                z_full = z_full + z_original * self.residual_scale
            z_full = flow.inverse(z_full)

        x_reconstructed = z_full * self.proj_gate + z_full * (1 - self.proj_gate)
        return x_reconstructed

# 3.3. ODEFunc
class ODEFunc(nn.Module):
    def __init__(self, 
                 layer_norm, 
                 decay_factor=0.1,
                 adaptive_factor=0.01,    # Factor para adaptación
                 stability_factor=0.1,    # Factor para estabilidad
                 alpha=0.2):             # Factor para balance de no linealidades
        super(ODEFunc, self).__init__()
        self.layer_norm = layer_norm
        self.decay_factor = decay_factor    # Control temporal base
        self.adaptive_factor = adaptive_factor
        self.stability_factor = stability_factor
        self.alpha = alpha
        
    def nonlinear_transform(self, e):
        """
        Transformación no lineal mejorada que combina diferentes
        no linealidades para capturar diferentes aspectos de los embeddings
        """
        # Combinación de no linealidades complementarias
        tanh_term = torch.tanh(e)        # Saturación suave, buena para normalización
        gelu_term = F.gelu(e)            # Mejor comportamiento para gradientes
        
        # Combinación adaptativa de las transformaciones
        combined = (1 - self.alpha) * tanh_term + self.alpha * gelu_term
        
        return combined

    def forward(self, t, e):
        """
        Transforma embeddings con dinámica mejorada y transformaciones
        no lineales optimizadas, manteniendo el propósito original.
        
        Args:
            t: Variable temporal
            e: Embeddings a transformar
        """
        # 1. Término de decaimiento adaptativo
        magnitude = torch.norm(e, dim=-1, keepdim=True)
        adaptive_decay = self.decay_factor * (1 + self.adaptive_factor * magnitude)
        decay_term = -adaptive_decay * e
        
        # 2. Transformación no lineal mejorada
        transform_term = self.nonlinear_transform(e)
        
        # 3. Término de estabilidad para preservar información importante
        stability_term = self.stability_factor * e
        
        # Combinación de términos preservando la estructura original
        updated_e = decay_term + transform_term + stability_term
        
        return self.layer_norm(updated_e)

    def extra_repr(self) -> str:
        return (f'decay_factor={self.decay_factor}, '
                f'adaptive_factor={self.adaptive_factor}, '
                f'stability_factor={self.stability_factor}, '
                f'alpha={self.alpha}')

# 3.4. LiquidNeuron y AdaptiveLiquidNeuron
class LiquidNeuron(nn.Module):
    def __init__(self, hidden_size):
        super(LiquidNeuron, self).__init__()
        self.hidden_size = hidden_size
        self.W_rec = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.1)
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.tau = nn.Parameter(torch.ones(hidden_size))
        self.decay = nn.Parameter(torch.ones(hidden_size))
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, t, h, e):
        dh_dt = (-self.decay * h + torch.matmul(e, self.W_rec.T) + self.bias) / self.tau
        dh_dt = self.layer_norm(dh_dt)
        return dh_dt

class AdaptiveLiquidNeuron(nn.Module):
    def __init__(self, hidden_size, dropout_rate=0.1):
        super(AdaptiveLiquidNeuron, self).__init__()
        self.base_params = LiquidNeuron(hidden_size)
        self.context_encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # Dropout después de ReLU
            nn.Linear(hidden_size, hidden_size)
        )
        self.param_modulator = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, t, h, e):
        # Codificación del contexto actual con dropout
        context = self.dropout(self.context_encoder(h))
        
        # Modulación adaptativa de parámetros con dropout
        param_adjustment = self.dropout(self.param_modulator(context))
        
        # Ajuste dinámico de parámetros base
        # Expande W_rec a [1, hidden_size, hidden_size] para broadcast
        W_rec_expanded = self.base_params.W_rec.unsqueeze(0)  # [1, hidden_size, hidden_size]
        
        # Ajusta W_rec por cada elemento del batch
        # [batch_size, hidden_size, 1] para multiplicación
        scaling_factor = (1 + param_adjustment).unsqueeze(2)  # [batch_size, hidden_size, 1]
        adjusted_W = W_rec_expanded * scaling_factor  # [batch_size, hidden_size, hidden_size]
        
        # Ajusta tau dinámicamente
        tau_adjusted = self.base_params.tau * torch.sigmoid(param_adjustment)  # [batch_size, hidden_size]
        
        # Calcula torch.bmm para multiplicar e con adjusted_W
        # e: [batch_size, hidden_size]
        # e_unsqueezed: [batch_size, 1, hidden_size]
        e_unsqueezed = e.unsqueeze(1)  # [batch_size, 1, hidden_size]
        
        # Transponer adjusted_W para [batch_size, hidden_size, hidden_size]
        adjusted_W_transposed = adjusted_W.transpose(2, 1)  # [batch_size, hidden_size, hidden_size]
        
        # Resultado de la multiplicación: [batch_size, 1, hidden_size]
        matmul_result = torch.bmm(e_unsqueezed, adjusted_W_transposed).squeeze(1)  # [batch_size, hidden_size]
        
        # Calcula dh_dt con los parámetros ajustados
        dh_dt = (-self.base_params.decay * h + matmul_result + self.base_params.bias) / tau_adjusted  # [batch_size, hidden_size]
        
        dh_dt = self.base_params.layer_norm(dh_dt)
        return dh_dt

# 3.5. ImprovedLiquidTimeCell
class ImprovedLiquidTimeCell(nn.Module):
    def __init__(self, hidden_size, dropout_rate=0.1):
        super(ImprovedLiquidTimeCell, self).__init__()
        self.neuron = AdaptiveLiquidNeuron(hidden_size, dropout_rate)
        self.warping_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout_rate),  # Dropout después de la activación
            nn.Linear(hidden_size, 1),
            nn.Softplus()
        )
        self.stability_gate = nn.Parameter(torch.zeros(1))
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, e, h, t_span):
        def ode_func(t, h):
            # Warping factor adaptativo basado en el estado actual
            warp = self.warping_layer(h)
            t_warped = t * warp

            dh = self.neuron(t_warped, h, e)
            stability_factor = torch.sigmoid(self.stability_gate)
            return dh * stability_factor

        h_next = odeint(
            ode_func,
            h,
            t_span,
            method='dopri5',
            adjoint_params=self.parameters(),
            rtol=1e-3,
            atol=1e-3
        )[-1]

        return h_next

# 3.6. AsyncLiquidCell
class AsyncLiquidCell(nn.Module):
    def __init__(self, hidden_size, dropout_rate=0.1):
        super(AsyncLiquidCell, self).__init__()
        self.time_embed = nn.Linear(1, hidden_size)
        self.attention_query = nn.Linear(hidden_size, hidden_size)
        self.attention_key = nn.Linear(hidden_size, hidden_size)
        self.attention_value = nn.Linear(hidden_size, hidden_size)
        self.attention_softmax = nn.Softmax(dim=-1)
        self.liquid_cell = ImprovedLiquidTimeCell(hidden_size, dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x, timestamps, h=None):
        """
        x: [batch_size, seq_length, hidden_size]
        timestamps: [batch_size, seq_length]
        h: [batch_size, hidden_size]
        """
        if h is None:
            h = torch.zeros(x.size(0), x.size(2)).to(x.device)

        # Embedding temporal
        t_embed = self.dropout(self.time_embed(timestamps.unsqueeze(-1)))
        
        # Incorporar la información temporal en las entradas
        x = x + t_embed  # [batch_size, seq_length, hidden_size]

        # Incorporar dropout en la atención
        Q = self.dropout(self.attention_query(h)).unsqueeze(1)  # [batch_size, 1, hidden_size]
        K = self.dropout(self.attention_key(x))  # [batch_size, seq_length, hidden_size]
        V = self.dropout(self.attention_value(x))  # [batch_size, seq_length, hidden_size]

        # Cálculo de atención
        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(x.size(-1))  # [batch_size, 1, seq_length]
        attn_weights = self.attention_softmax(attn_scores)  # [batch_size, 1, seq_length]
        context = torch.bmm(attn_weights, V)  # [batch_size, 1, hidden_size]
        context = context.squeeze(1)  # [batch_size, hidden_size]

        # Actualizar el estado oculto utilizando la célula de tiempo líquido mejorada
        h_next = self.liquid_cell(context, h, torch.linspace(0, 1, steps=5).to(x.device))  # [batch_size, hidden_size]

        return h_next, attn_weights  # Retorna también los pesos de atención

# 3.7. OptimizedLiquidEmbeddingMFR
class OptimizedLiquidEmbeddingMFR(nn.Module):
    def __init__(self, vocab_size, embedding_dim, latent_dim, hidden_size, hidden_dim=128, 
                 num_flows=4, share_params=True, dropout_rate=0.1):
        super(OptimizedLiquidEmbeddingMFR, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding_dropout = nn.Dropout(dropout_rate)  # Dropout específico para embeddings
        self.dropout = nn.Dropout(dropout_rate)

        self.hidden_size = hidden_size
        self.latent_dim = latent_dim

        self.flow_reducer = OptimizedFlowDimensionalityReduction(
            original_dim=embedding_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_flows=num_flows,
            share_params=share_params
        )

        self.W_e = nn.Linear(latent_dim, hidden_size)
        self.bias_e = nn.Parameter(torch.zeros(hidden_size))
        self.transform_gate = nn.Parameter(torch.tensor(0.0))
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

        self.ode_func = ODEFunc(
            layer_norm=self.layer_norm,
            decay_factor=0.1,
            adaptive_factor=0.01,
            stability_factor=0.1,
            alpha=0.2
        )

    def forward(self, x, attention_mask, t_span):
        B, S = x.size()
        embedded = self.embedding_dropout(self.embedding(x))

        embedded = embedded.view(B, S, -1)  # Asegurar la forma

        z, log_det_jacobian = self.flow_reducer(embedded)  # [batch_size, seq_length, latent_dim]

        if self.latent_dim == self.hidden_size:
            e0 = self.W_e(z) * self.transform_gate + z * (1 - self.transform_gate)
        else:
            e0 = self.W_e(z)
        e0 = e0 + self.bias_e
        e0 = self.layer_norm(e0)
        e0 = self.dropout(e0)

        e_liquid = odeint(
            self.ode_func,
            e0,
            t_span,
            method='dopri5',
            adjoint_params=self.parameters(),
            rtol=1e-3,
            atol=1e-3
        )[-1]  # [batch_size, seq_length, hidden_size]
        e_liquid = e_liquid + e0 * 0.1  # Residual connection

        return e_liquid, log_det_jacobian

# 3.8. LiquidTextGenerationModel
class LiquidTextGenerationModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, latent_dim, hidden_size, hidden_dim=128, 
                 num_flows=4, share_params=True, dropout_rate=0.1):
        super(LiquidTextGenerationModel, self).__init__()
        
        # Embedding Layer
        self.liquid_embedding = OptimizedLiquidEmbeddingMFR(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            latent_dim=latent_dim,
            hidden_size=hidden_size,
            hidden_dim=hidden_dim,
            num_flows=num_flows,
            share_params=share_params,
            dropout_rate=dropout_rate
        )
        
        # Liquid Cell para procesamiento secuencial
        self.liquid_cell = AsyncLiquidCell(hidden_size, dropout_rate)
        
        # Proyección a vocabulario para generación de tokens
        self.output_proj = nn.Linear(hidden_size, vocab_size)
        
        # Dropout adicional
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attention_mask, t_span, timestamps, h=None):
        # Aplicar dropout en diferentes etapas del forward pass
        e_liquid, log_det_jacobian = self.liquid_embedding(x, attention_mask, t_span)
        e_liquid = self.dropout(e_liquid)
        
        h, attn_weights = self.liquid_cell(e_liquid, timestamps, h)
        h = self.dropout(h)
        
        # Proyección a vocabulario para obtener logits de tokens
        logits = self.output_proj(e_liquid)  # [batch_size, seq_length, vocab_size]
        
        return logits, h, log_det_jacobian, attn_weights

# ------------------------------------
# 4. Definición del Scheduler de Calentamiento
# ------------------------------------
class WarmupScheduler:
    def __init__(self, optimizer, warmup_steps, base_lr):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.current_step = 0
        
    def step(self):
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.current_step += 1
        
    def get_lr(self):
        if self.current_step < self.warmup_steps:
            return self.base_lr * (self.current_step + 1) / self.warmup_steps
        return self.base_lr

# ------------------------------------
# 5. Función para Generar Timestamps
# ------------------------------------
def generate_timestamps(batch_size, seq_length, device):
    # Generar timestamps uniformemente espaciados entre 0 y 1
    timestamps = torch.zeros(batch_size, seq_length).to(device)
    for i in range(batch_size):
        event_points = random.sample(range(seq_length), k=max(1, seq_length // 10))  # Aproximadamente 10% eventos
        for point in event_points:
            timestamps[i, point] = random.uniform(0, 1)
    # Normalizar timestamps para que estén entre 0 y 1
    timestamps = torch.cumsum(timestamps, dim=1)
    # Evitar división por cero
    last_vals = timestamps[:, -1].unsqueeze(1).clamp(min=1e-5)
    timestamps = timestamps / last_vals
    return timestamps

# ------------------------------------
# 6. Aumentación de Datos Específica para Texto
# ------------------------------------
class TextAugmentation:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.mask_token_id = tokenizer.tokenizer.mask_token_id
        self.vocab_size = tokenizer.vocab_size

    def random_mask(self, input_ids, mask_prob=0.15):
        """Enmascara aleatoriamente tokens manteniendo la estructura sintáctica"""
        masked = input_ids.clone()
        mask = torch.rand(masked.shape) < mask_prob
        # Asegurarse de no enmascarar tokens especiales
        special_tokens = [self.tokenizer.pad_token_id, self.tokenizer.eos_token_id, self.tokenizer.bos_token_id]
        for token_id in special_tokens:
            mask = mask & (masked != token_id)
        masked[mask] = self.mask_token_id
        return masked

    def token_cutoff(self, input_ids, max_cut=0.1):
        """Recorta aleatoriamente tokens del final manteniendo coherencia"""
        cut_point = torch.randint(
            int(input_ids.size(1) * (1 - max_cut)),
            input_ids.size(1) + 1,
            (1,)
        ).item()
        return input_ids[:, :cut_point]

    def temporal_shuffle(self, input_ids, window_size=3):
        """Permuta localmente tokens manteniendo coherencia temporal"""
        shuffled = input_ids.clone()
        batch_size, seq_length = shuffled.size()
        
        for i in range(0, seq_length - window_size + 1, window_size):
            # Permutación local dentro de la ventana
            window = shuffled[:, i:i+window_size].clone()
            perm = torch.randperm(window_size)
            shuffled[:, i:i+window_size] = window[:, perm]
            
        return shuffled

# ------------------------------------
# 7. Regularización Específica para Texto
# ------------------------------------
class TextSpecificRegularization(nn.Module):
    def __init__(self, hidden_size):
        super(TextSpecificRegularization, self).__init__()
        self.hidden_size = hidden_size
        
        # Parámetros para diferentes aspectos de regularización
        self.semantic_gate = nn.Parameter(torch.ones(1))
        self.temporal_gate = nn.Parameter(torch.ones(1))
        self.structure_gate = nn.Parameter(torch.ones(1))

    def semantic_coherence_loss(self, embeddings):
        """Mantiene coherencia semántica entre tokens cercanos"""
        # Similitud coseno entre embeddings adyacentes
        norm_embeddings = F.normalize(embeddings, p=2, dim=-1)
        similarity = torch.bmm(norm_embeddings, norm_embeddings.transpose(1, 2))
        
        # Penaliza cambios bruscos en la similitud
        temporal_diff = torch.diff(similarity, dim=1)
        coherence_loss = torch.mean(torch.abs(temporal_diff))
        
        return coherence_loss * torch.sigmoid(self.semantic_gate)

    def temporal_consistency_loss(self, hidden_states):
        """Mantiene consistencia temporal en las representaciones"""
        # Diferencias temporales en estados ocultos
        temporal_diff = torch.diff(hidden_states, dim=1)
        consistency_loss = torch.mean(torch.norm(temporal_diff, dim=-1))
        
        return consistency_loss * torch.sigmoid(self.temporal_gate)

    def structural_preservation_loss(self, attention_weights):
        """Preserva estructura jerárquica en patrones de atención"""
        # Entropía de patrones de atención
        entropy = -torch.sum(attention_weights * torch.log(attention_weights + 1e-10), dim=-1)
        structure_loss = torch.mean(entropy)
        
        return structure_loss * torch.sigmoid(self.structure_gate)

# ------------------------------------
# 8. Modificación de la Función de Pérdida Principal
# ------------------------------------
class TextAwareTrainingLoss(nn.Module):
    def __init__(self, base_criterion):
        super(TextAwareTrainingLoss, self).__init__()
        self.base_criterion = base_criterion
        self.semantic_weight = nn.Parameter(torch.tensor(0.1))
        self.temporal_weight = nn.Parameter(torch.tensor(0.1))
        self.structural_weight = nn.Parameter(torch.tensor(0.1))

    def forward(self, logits, targets, embeddings, hidden_states, attention_weights, regularizer):
        # Aplanar las salidas y etiquetas para calcular la pérdida de entropía cruzada
        # logits: [batch_size, seq_length, vocab_size] -> [batch_size * seq_length, vocab_size]
        # targets: [batch_size, seq_length] -> [batch_size * seq_length]
        logits = logits.view(-1, logits.size(-1))
        targets = targets.view(-1)
        
        # Pérdida base de generación de texto
        base_loss = self.base_criterion(logits, targets)
        
        # Pérdidas de regularización específicas para texto
        semantic_loss = regularizer.semantic_coherence_loss(embeddings)
        temporal_loss = regularizer.temporal_consistency_loss(hidden_states)
        structural_loss = regularizer.structural_preservation_loss(attention_weights)
        
        # Combinación ponderada adaptativa
        total_loss = base_loss + \
                    torch.sigmoid(self.semantic_weight) * semantic_loss + \
                    torch.sigmoid(self.temporal_weight) * temporal_loss + \
                    torch.sigmoid(self.structural_weight) * structural_loss
                    
        return total_loss, {
            'base_loss': base_loss.item(),
            'semantic_loss': semantic_loss.item(),
            'temporal_loss': temporal_loss.item(),
            'structural_loss': structural_loss.item()
        }

# ------------------------------------
# 9. Funciones de Entrenamiento y Evaluación
# ------------------------------------
def evaluate_model(model, val_loader, criterion, regularizer, device):
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0

    # Barra de progreso para la validación
    val_loader_tqdm = tqdm(val_loader, desc=f"Validación", leave=False)
    with torch.no_grad():
        for input_ids, attention_mask, labels in val_loader_tqdm:
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            
            # Generar timestamps
            timestamps = generate_timestamps(input_ids.size(0), input_ids.size(1), device)
            
            # Forward pass
            logits, hidden, log_det_jacobian, attn_weights = model(
                input_ids, 
                attention_mask, 
                torch.linspace(0, 1, steps=5).to(device), 
                timestamps
            )
            
            embeddings = model.liquid_embedding.embedding(input_ids)
            loss, _ = criterion(
                logits,
                labels,
                embeddings,
                hidden,
                attn_weights,
                regularizer
            )
            val_loss += loss.item()
            
            # Calcula precisión solo donde los labels no son pad
            active = labels != model.liquid_embedding.tokenizer.pad_token_id
            active_logits = logits[active]
            active_labels = labels[active]
            _, predicted = active_logits.max(1)
            val_correct += predicted.eq(active_labels).sum().item()
            val_total += active_labels.size(0)

            # Actualizar la barra de progreso con la pérdida y precisión actuales
            current_val_loss = loss.item()
            current_val_accuracy = 100. * val_correct / val_total if val_total > 0 else 0
            val_loader_tqdm.set_postfix({'Pérdida': f"{current_val_loss:.4f}", 'Precisión': f"{current_val_accuracy:.2f}%"})

    val_accuracy = 100. * val_correct / val_total if val_total > 0 else 0
    avg_val_loss = val_loss / len(val_loader)

    # Imprimir los resultados de la evaluación
    print(f"Pérdida validación: {avg_val_loss:.4f}, Precisión validación: {val_accuracy:.2f}%")
    print("-" * 50)

def train_with_augmentation(model, train_loader, val_loader, tokenizer, device, num_epochs=10, learning_rate=1e-4):
    augmenter = TextAugmentation(tokenizer)
    text_regularizer = TextSpecificRegularization(model.liquid_embedding.hidden_size).to(device)
    criterion = TextAwareTrainingLoss(nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)).to(device)
    
    optimizer = optim.AdamW([
        {'params': model.parameters()},
        {'params': text_regularizer.parameters(), 'lr': learning_rate},
        {'params': criterion.parameters(), 'lr': learning_rate}
    ], lr=learning_rate, weight_decay=5e-3)
    
    warmup_scheduler = WarmupScheduler(optimizer, warmup_steps=1000, base_lr=learning_rate)
    decay_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        
    for epoch in range(num_epochs):
        model.train()
        text_regularizer.train()
        criterion.train()
        
        total_loss = 0
        correct = 0
        total = 0

        # Barra de progreso para el entrenamiento
        train_loader_tqdm = tqdm(train_loader, desc=f"Entrenamiento Época {epoch+1}/{num_epochs}", leave=False)
        for batch_idx, (input_ids, attention_mask, labels) in enumerate(train_loader_tqdm):
            optimizer.zero_grad()
            
            # Aplicar aumentación de datos
            if random.random() < 0.5:
                input_ids = augmenter.random_mask(input_ids)
            if random.random() < 0.3:
                input_ids = augmenter.temporal_shuffle(input_ids)
            # Nota: token_cutoff no se usa aquí, pero se puede incluir si se desea
            
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            
            # Generar timestamps
            timestamps = generate_timestamps(input_ids.size(0), input_ids.size(1), device)
            
            # Forward pass
            logits, hidden, log_det_jacobian, attn_weights = model(
                input_ids, 
                attention_mask, 
                torch.linspace(0, 1, steps=5).to(device), 
                timestamps
            )
            
            embeddings = model.liquid_embedding.embedding(input_ids)
            
            loss, loss_components = criterion(
                logits,
                labels,
                embeddings,
                hidden,
                attn_weights,
                text_regularizer
            )
            
            # Backward y optimización
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            warmup_scheduler.step()
            
            total_loss += loss.item()
            # Calcula precisión solo donde los labels no son pad
            active = labels != tokenizer.pad_token_id
            active_logits = logits[active]
            active_labels = labels[active]
            _, predicted = active_logits.max(1)
            correct += predicted.eq(active_labels).sum().item()
            total += active_labels.size(0)
            
            # Actualizar la barra de progreso con la pérdida y precisión actuales
            current_loss = loss.item()
            current_accuracy = 100. * correct / total if total > 0 else 0
            train_loader_tqdm.set_postfix({'Pérdida': f"{current_loss:.4f}", 'Precisión': f"{current_accuracy:.2f}%"})

        train_accuracy = 100. * correct / total if total > 0 else 0
        avg_loss = total_loss / len(train_loader)

        # Imprimir los resultados de la época
        print(f"Época {epoch+1}/{num_epochs}")
        print(f"Pérdida entrenamiento: {avg_loss:.4f}, Precisión entrenamiento: {train_accuracy:.2f}%")
        
        # Actualizar el scheduler de decaimiento
        decay_scheduler.step()
        
        # Evaluación y métricas al final de cada época
        evaluate_model(model, val_loader, criterion, text_regularizer, device)

    return model

# ------------------------------------
# 10. Función de Generación de Texto
# ------------------------------------
def generate_text(model, tokenizer, prompt, max_length=50, device='cuda', temperature=1.0, top_k=50, top_p=0.95):
    model.eval()
    generated = []
    with torch.no_grad():
        input_ids, attention_mask = tokenizer.tokenize([prompt], max_length=512, padding='max_length', truncation=True)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        # Generar timestamps
        timestamps = generate_timestamps(input_ids.size(0), input_ids.size(1), device)
        
        h = None
        for _ in range(max_length):
            logits, h, log_det_jacobian, attn_weights = model(
                input_ids, 
                attention_mask, 
                torch.linspace(0, 1, steps=5).to(device), 
                timestamps,
                h
            )
            logits = logits[:, -1, :] / temperature  # Tomar logits del último token
            # Aplicar top-k y top-p (nucleus) sampling
            filtered_logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
            probabilities = F.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            generated.append(next_token.item())
            # Actualizar input_ids y attention_mask
            input_ids = torch.cat([input_ids, next_token], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=1)
            timestamps = generate_timestamps(input_ids.size(0), input_ids.size(1), device)
        
        generated_text = tokenizer.decode(generated, skip_special_tokens=True)
        return generated_text

def top_k_top_p_filtering(logits, top_k=50, top_p=0.95, filter_value=-float('Inf')):
    """Filtros para top-k y nucleus (top-p) sampling."""
    # Top-K filtering
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))  # Guardar top_k dentro del rango
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    # Top-P (nucleus) filtering
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Eliminar tokens con probabilidad acumulada > top_p
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift para incluir el primer token que excede top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits

# ------------------------------------
# 11. Función Principal Actualizada
# ------------------------------------
def main():
    batch_size = 32
    embedding_dim = 512
    latent_dim = 128
    hidden_size = 256
    hidden_dim = 128
    num_flows = 4
    num_epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # Inicializar el Tokenizador de LED
    tokenizer = LEDTokenizerWrapper(pretrained_model_name='allenai/led-base-16384')

    # Preparar DataLoaders con un máximo de 10000 ejemplos
    train_loader, val_loader, vocab_size = create_dataloaders(
        batch_size=batch_size,
        tokenizer=tokenizer,
        max_length=256,
        max_examples=10000  # Ajusta según tus recursos
    )

    # Crear el Modelo de Generación de Texto
    model = LiquidTextGenerationModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        latent_dim=latent_dim,
        hidden_size=hidden_size,
        hidden_dim=hidden_dim,
        num_flows=num_flows,
        share_params=True,
        dropout_rate=0.1
    )

    # Mover el modelo al dispositivo
    model = model.to(device)

    # Entrenar y Evaluar con aumentación y regularización
    model = train_with_augmentation(model, train_loader, val_loader, tokenizer, device, num_epochs=num_epochs, learning_rate=1e-4)

    # Guardar el modelo entrenado
    torch.save(model.state_dict(), 'liquid_text_generation_model.pth')
    print("Modelo guardado exitosamente.")

    return model, train_loader, val_loader

if __name__ == "__main__":
    try:
        print("PyTorch está disponible:", torch.cuda.is_available())
        model, train_loader, val_loader = main()
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
