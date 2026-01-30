import numpy as np
import matplotlib.pyplot as plt
import cv2
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap

# ===============================
# CONFIGURAÇÃO: bits fixos ou aleatórios
# ===============================
print("SIMULAÇÃO DE PROPAGAÇÃO DE ONDAS ELETROMAGNÉTICAS")
print()

modo = input("choose fix or random: ").strip().lower()

if modo == 'fix':
    BIT_MODE = 'fix'
    bits = np.array([0, 1, 1, 0, 0, 1, 0])
    print(f"fix chosen")
elif modo == 'random':
    BIT_MODE = 'rand'
    bits = np.random.randint(0, 2, 8)
    print(f"random chosen")
else:
    # Entrada válida por padrão (assume fix)
    BIT_MODE = 'fix'
    bits = np.array([0, 1, 1, 0, 0, 1, 0])
    print(f"fix chosen")

print()

# ===============================
# Parâmetros da simulação
# ===============================
Nx, Ny = 200, 120
steps = 2400
c = 1.0
dt = 0.5
dx = 1.0

# ===============================
# Campos elétricos
# ===============================
E_prev = np.zeros((Nx, Ny))
E = np.zeros((Nx, Ny))
E_next = np.zeros((Nx, Ny))

# ===============================
# Obstáculos (1 = parede, 0 = livre)
# ===============================
walls = np.zeros((Nx, Ny))
wall_x = Nx // 2
walls[wall_x, :] = 1
walls[wall_x, 50:70] = 0  # abertura

# ===============================
# Sequência binária e modulação AM
# ===============================
bit_rate = 300
bit_signal = []

for bit in bits:
    if bit == 1:
        bit_signal.extend([1] * bit_rate)
    else:
        bit_signal.extend([0] * bit_rate)

bit_signal = np.array(bit_signal)

fc = 0.12
t = np.arange(len(bit_signal))
carrier = np.sin(2 * np.pi * fc * t)
am_signal = bit_signal * carrier * 1.5

# ===============================
# Fonte e receptor
# ===============================
source_x, source_y = 30, Ny // 2
ep_x, ep_y = 140, Ny // 2
ep_signal = []

# ===============================
# Vídeo
# ===============================
video_width = 1280
video_height = 720

video = cv2.VideoWriter(
    "propagacao-720-final.avi",
    cv2.VideoWriter_fourcc(*"XVID"),
    30,
    (video_width, video_height),
    True
)

# ===============================
# Colormap melhorado - SEM FUNDO ABACATE
# ===============================
colors = ['#000033', '#0000FF', '#00FFFF', '#FFFFFF', '#FFFF00', '#FF0000']
cmap_custom = LinearSegmentedColormap.from_list('custom', colors, N=256)

E_min_fixed = -0.5
E_max_fixed = 0.5

# ===============================
# Simulação FDTD 2D
# ===============================
for n in range(steps):

    # Laplaciano 2D
    laplacian = (
        np.roll(E, 1, axis=0) +
        np.roll(E, -1, axis=0) +
        np.roll(E, 1, axis=1) +
        np.roll(E, -1, axis=1) -
        4 * E
    )

    E_next = (
        2 * E - E_prev +
        (c * dt / dx) ** 2 * laplacian
    )

    # Aplicar paredes (reflexão)
    E_next[walls == 1] = 0

    # Condições de contorno (bordas)
    E_next[0, :] = 0
    E_next[-1, :] = 0
    E_next[:, 0] = 0
    E_next[:, -1] = 0

    # Fonte AM
    if n < len(am_signal):
        E_next[source_x, source_y] += am_signal[n]

    # *** CALCULAR Ep(t) - VALOR DO CAMPO NO PONTO RECEPTOR ***
    Ep = E[ep_x, ep_y]
    ep_signal.append(Ep)

    # Frame do vídeo com colormap melhorado
    E_norm = np.clip((E - E_min_fixed) / (E_max_fixed - E_min_fixed), 0, 1)
    E_norm = np.power(E_norm, 0.7)
    
    frame_colored = cmap_custom(E_norm.T)
    frame_bgr = (frame_colored[:, :, :3] * 255).astype(np.uint8)
    frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_RGB2BGR)

    frame_mid = cv2.resize(
        frame_bgr,
        (Nx * 4, Ny * 4),
        interpolation=cv2.INTER_LINEAR
    )

    frame_720p = cv2.resize(
        frame_mid,
        (video_width, video_height),
        interpolation=cv2.INTER_LINEAR
    )

    # Adicionar marcadores
    source_scaled = (int(source_x * 4 * video_width / (Nx*4)), 
                     int(source_y * 4 * video_height / (Ny*4)))
    cv2.circle(frame_720p, source_scaled, 20, (0, 0, 255), 3)
    
    ep_scaled = (int(ep_x * 4 * video_width / (Nx*4)), 
                 int(ep_y * 4 * video_height / (Ny*4)))
    cv2.circle(frame_720p, ep_scaled, 20, (0, 255, 0), 3)
    
    # Desenhar parede preta com abertura
    wall_x_scaled = int(wall_x * 4 * video_width / (Nx*4))
    opening_start_scaled = int(50 * 4 * video_height / (Ny*4))
    opening_end_scaled = int(70 * 4 * video_height / (Ny*4))
    
    cv2.line(frame_720p, (wall_x_scaled, 0), 
             (wall_x_scaled, opening_start_scaled), (0, 0, 0), 5)
    
    cv2.line(frame_720p, (wall_x_scaled, opening_end_scaled), 
             (wall_x_scaled, video_height), (0, 0, 0), 5)

    video.write(frame_720p)

    # Progress bar
    if n % 100 == 0:
        progress = (n / steps) * 100
        print(f"progresso: {progress:.0f}%", end='\r')

    # Avançar no tempo
    E_prev, E = E, E_next

video.release()
print()  # Nova linha após progresso

# ===============================
# Plot 1 – Sequência Binária
# ===============================

plt.figure(figsize=(16, 3))
time_axis = np.arange(len(bits)) * bit_rate
plt.step(time_axis, bits, where='post', linewidth=2.5, color='blue')

plt.ylim(-0.2, 1.3)
plt.xlim(-50, len(bits) * bit_rate + 50)

if BIT_MODE == 'fix':
    titulo = f"Sequência Binária - Padrão: {''.join(map(str, bits))}"
else:
    titulo = f"Sequência Binária - Aleatório: {''.join(map(str, bits))}"

plt.title(titulo, fontsize=14, fontweight='bold')
plt.ylabel("Bit", fontsize=12)
plt.xlabel("Tempo (amostras)", fontsize=12)
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig("01_bits_padrao.png", dpi=300, bbox_inches='tight')

# ===============================
# Plot 2 – Sinal AM
# ===============================
plt.figure(figsize=(16, 3))
time_samples = np.arange(len(am_signal))
plt.plot(time_samples, am_signal, linewidth=0.8, color='blue')
plt.title("Sinal Modulado em AM (Fonte)", fontsize=14, fontweight='bold')
plt.ylabel("Amplitude", fontsize=12)
plt.xlabel("Tempo (amostras)", fontsize=12)
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig("02_am_sinal.png", dpi=300, bbox_inches='tight')

# ===============================
# Plot 3 – Campo Elétrico Ep(t) no Receptor
# ===============================
plt.figure(figsize=(16, 4))

ep_signal_array = np.array(ep_signal)
time_ep = np.arange(len(ep_signal_array))

plt.plot(time_ep, ep_signal_array, linewidth=0.6, alpha=0.9, color='blue')

# Adicionar envelope para visualização
window_size = 50
envelope = np.convolve(np.abs(ep_signal_array), 
                       np.ones(window_size)/window_size, 
                       mode='same')
plt.plot(time_ep, envelope, linewidth=2, color='red', 
         label='Envelope |Ep(t)|', alpha=0.7)
plt.plot(time_ep, -envelope, linewidth=2, color='red', alpha=0.7)

plt.title("Campo Elétrico Ep(t) no Ponto Receptor", 
          fontsize=14, fontweight='bold')
plt.ylabel("Ep (unidades arbitrárias)", fontsize=12)
plt.xlabel("Tempo (amostras)", fontsize=12)
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(fontsize=11, loc='upper right')
plt.tight_layout()
plt.savefig("03_ep_receptor.png", dpi=300, bbox_inches='tight')

# ===============================
# Salvar dados de Ep(t) em arquivo texto
# ===============================
np.savetxt("ep_dados.txt", ep_signal_array, 
           header=f"Campo Elétrico Ep(t) no receptor\n"
                  f"Modo: {BIT_MODE}\n"
                  f"Bits: {''.join(map(str, bits))}\n"
                  f"Posição: ({ep_x}, {ep_y})\n"
                  f"Total de amostras: {len(ep_signal_array)}",
           fmt='%.6f')

# ===============================
# Estatísticas de Ep
# ===============================
print("ESTATÍSTICAS DO CAMPO ELÉTRICO Ep(t)")
print(f"Valor máximo:    {np.max(ep_signal_array):>10.4f}")
print(f"Valor mínimo:    {np.min(ep_signal_array):>10.4f}")
print(f"Amplitude média: {np.mean(np.abs(ep_signal_array)):>10.4f}")
print(f"Desvio padrão:   {np.std(ep_signal_array):>10.4f}")
print(f"Root Mean Square: {np.sqrt(np.mean(ep_signal_array**2)):>10.4f}")

# Mostrar plots por 20 segundos
plt.show(block=False)
plt.pause(20)
plt.close('all')
