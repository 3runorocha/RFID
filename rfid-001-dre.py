import numpy as np
import matplotlib.pyplot as plt
import cv2
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap

# CONFIGURAÇÃO INICIAL
def configurar_modo():
    """Escolhe entre modo fixo ou aleatório para sequência de bits"""
    print("SIMULAÇÃO DE PROPAGAÇÃO DE ONDAS ELETROMAGNÉTICAS")
    print()
    
    modo = input("choose fix or random: ").strip().lower()
    
    if modo == 'fix':
        bits = np.array([0, 1, 1, 0, 0, 1, 0])
        print(f"fix chosen")
    elif modo == 'random':
        bits = np.random.randint(0, 2, 8)
        print(f"random chosen")
    else:
        bits = np.array([0, 1, 1, 0, 0, 1, 0])
        print(f"fix chosen")
    
    print()
    return bits, modo

# GEOMETRIA E OBSTÁCULOS
def criar_geometria(Nx, Ny):
    """Cria grade de simulação com parede e abertura para difração"""
    walls = np.zeros((Nx, Ny))
    wall_x = Nx // 2
    
    walls[wall_x, :] = 1          # Parede sólida
    walls[wall_x, 50:70] = 0      # Abertura (fenda) para difração
    
    return walls, wall_x

# MODULAÇÃO DO SINAL
def gerar_sinal_am(bits, bit_rate=300, fc=0.12, amplitude=1.5):
    """Gera sinal AM a partir dos bits digitais"""
    bit_signal = []
    
    for bit in bits:
        if bit == 1:
            bit_signal.extend([1] * bit_rate)
        else:
            bit_signal.extend([0] * bit_rate)
    
    bit_signal = np.array(bit_signal)
    
    t = np.arange(len(bit_signal))
    carrier = np.sin(2 * np.pi * fc * t)      # Portadora senoidal
    am_signal = bit_signal * carrier * amplitude  # Modulação AM
    
    return bit_signal, am_signal

# SIMULAÇÃO FDTD
def simular_propagacao(Nx, Ny, steps, am_signal, walls, source_pos, ep_pos):
    """Simula propagação de ondas EM usando método FDTD"""
    
    # Parâmetros físicos
    c = 1.0      # Velocidade da luz
    dt = 0.5     # Passo temporal
    dx = 1.0     # Passo espacial
    
    # Campos elétricos (tempo anterior, atual, próximo)
    E_prev = np.zeros((Nx, Ny))
    E = np.zeros((Nx, Ny))
    E_next = np.zeros((Nx, Ny))
    
    ep_signal = []
    source_x, source_y = source_pos
    ep_x, ep_y = ep_pos
    
    # Setup do vídeo
    video = cv2.VideoWriter(
        "propagacao-720-final.avi",
        cv2.VideoWriter_fourcc(*"XVID"),
        30,
        (1280, 720),
        True
    )
    
    colors = ['#000033', '#0000FF', '#00FFFF', '#FFFFFF', '#FFFF00', '#FF0000']
    cmap_custom = LinearSegmentedColormap.from_list('custom', colors, N=256)
    
    # Loop principal da simulação
    for n in range(steps):
        
        # Equação de onda 2D: ∇²E = (1/c²)∂²E/∂t²
        laplacian = (
            np.roll(E, 1, axis=0) +
            np.roll(E, -1, axis=0) +
            np.roll(E, 1, axis=1) +
            np.roll(E, -1, axis=1) -
            4 * E
        )
        
        # Atualização temporal (método de diferenças finitas)
        E_next = 2 * E - E_prev + (c * dt / dx) ** 2 * laplacian
        
        # Condições de contorno: paredes refletem ondas
        E_next[walls == 1] = 0
        
        # Bordas
        E_next[0, :] = 0
        E_next[-1, :] = 0
        E_next[:, 0] = 0
        E_next[:, -1] = 0
        
        # Fonte emissora (injeta sinal AM)
        if n < len(am_signal):
            E_next[source_x, source_y] += am_signal[n]
        
        # Medir campo elétrico no ponto receptor (Ep)
        Ep = E[ep_x, ep_y]
        ep_signal.append(Ep)
        
        # Renderizar frame do vídeo
        renderizar_frame(E, cmap_custom, video, source_pos, ep_pos, walls, Nx, Ny)
        
        if n % 100 == 0:
            progress = (n / steps) * 100
            print(f"progresso: {progress:.0f}%", end='\r')
        
        # Avançar no tempo
        E_prev, E = E, E_next
    
    video.release()
    print()
    
    return np.array(ep_signal)

def renderizar_frame(E, cmap, video, source_pos, ep_pos, walls, Nx, Ny):
    """Renderiza um frame da simulação para o vídeo"""
    E_norm = np.clip((E - (-0.5)) / (0.5 - (-0.5)), 0, 1)
    E_norm = np.power(E_norm, 0.7)
    
    frame_colored = cmap(E_norm.T)
    frame_bgr = (frame_colored[:, :, :3] * 255).astype(np.uint8)
    frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_RGB2BGR)
    
    frame_mid = cv2.resize(frame_bgr, (Nx * 4, Ny * 4), interpolation=cv2.INTER_LINEAR)
    frame_720p = cv2.resize(frame_mid, (1280, 720), interpolation=cv2.INTER_LINEAR)
    
    source_x, source_y = source_pos
    ep_x, ep_y = ep_pos
    
    source_scaled = (int(source_x * 4 * 1280 / (Nx*4)), int(source_y * 4 * 720 / (Ny*4)))
    cv2.circle(frame_720p, source_scaled, 20, (0, 0, 255), 3)
    
    ep_scaled = (int(ep_x * 4 * 1280 / (Nx*4)), int(ep_y * 4 * 720 / (Ny*4)))
    cv2.circle(frame_720p, ep_scaled, 20, (0, 255, 0), 3)
    
    wall_x = Nx // 2
    wall_x_scaled = int(wall_x * 4 * 1280 / (Nx*4))
    opening_start = int(50 * 4 * 720 / (Ny*4))
    opening_end = int(70 * 4 * 720 / (Ny*4))
    
    cv2.line(frame_720p, (wall_x_scaled, 0), (wall_x_scaled, opening_start), (0, 0, 0), 5)
    cv2.line(frame_720p, (wall_x_scaled, opening_end), (wall_x_scaled, 720), (0, 0, 0), 5)
    
    video.write(frame_720p)

# PLOTAGEM
def plotar_sequencia_binaria(bits, bit_rate, modo):
    """Plota a sequência binária original"""
    plt.figure(figsize=(16, 3))
    time_axis = np.arange(len(bits)) * bit_rate
    plt.step(time_axis, bits, where='post', linewidth=2.5, color='blue')
    
    plt.ylim(-0.2, 1.3)
    plt.xlim(-50, len(bits) * bit_rate + 50)
    
    if modo == 'fix':
        titulo = f"Sequência Binária - Padrão: {''.join(map(str, bits))}"
    else:
        titulo = f"Sequência Binária - Aleatório: {''.join(map(str, bits))}"
    
    plt.title(titulo, fontsize=14, fontweight='bold')
    plt.ylabel("Bit", fontsize=12)
    plt.xlabel("Tempo (amostras)", fontsize=12)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig("01_bits_padrao.png", dpi=300, bbox_inches='tight')

def plotar_sinal_am(am_signal):
    """Plota o sinal modulado em AM"""
    plt.figure(figsize=(16, 3))
    time_samples = np.arange(len(am_signal))
    plt.plot(time_samples, am_signal, linewidth=0.8, color='blue')
    plt.title("Sinal Modulado em AM (Fonte)", fontsize=14, fontweight='bold')
    plt.ylabel("Amplitude", fontsize=12)
    plt.xlabel("Tempo (amostras)", fontsize=12)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig("02_am_sinal.png", dpi=300, bbox_inches='tight')

def plotar_campo_receptor(ep_signal):
    """Plota o campo elétrico medido no receptor com envelope"""
    plt.figure(figsize=(16, 4))
    
    time_ep = np.arange(len(ep_signal))
    plt.plot(time_ep, ep_signal, linewidth=0.6, alpha=0.9, color='blue')
    
    # Envelope (detecta amplitude modulada)
    window_size = 50
    envelope = np.convolve(np.abs(ep_signal), 
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

def salvar_dados(ep_signal, bits, modo, ep_pos):
    """Salva dados numéricos de Ep em arquivo texto"""
    ep_x, ep_y = ep_pos
    np.savetxt("ep_dados.txt", ep_signal, 
               header=f"Campo Elétrico Ep(t) no receptor\n"
                      f"Modo: {modo}\n"
                      f"Bits: {''.join(map(str, bits))}\n"
                      f"Posição: ({ep_x}, {ep_y})\n"
                      f"Total de amostras: {len(ep_signal)}",
               fmt='%.6f')

def exibir_estatisticas(ep_signal):
    """Calcula e exibe estatísticas do campo elétrico"""
    print("ESTATÍSTICAS DO CAMPO ELÉTRICO Ep(t)")
    print(f"Valor máximo:    {np.max(ep_signal):>10.4f}")
    print(f"Valor mínimo:    {np.min(ep_signal):>10.4f}")
    print(f"Amplitude média: {np.mean(np.abs(ep_signal)):>10.4f}")
    print(f"Desvio padrão:   {np.std(ep_signal):>10.4f}")
    print(f"RMS: {np.sqrt(np.mean(ep_signal**2)):>10.4f}")

# PROGRAMA PRINCIPAL
if __name__ == "__main__":
    
    # Configuração
    bits, modo = configurar_modo()
    
    # Parâmetros da simulação
    Nx, Ny = 200, 120
    steps = 2400
    
    # Geometria
    walls, wall_x = criar_geometria(Nx, Ny)
    
    # Gerar sinal
    bit_signal, am_signal = gerar_sinal_am(bits)
    
    # Posições
    source_pos = (30, Ny // 2)
    ep_pos = (140, Ny // 2)
    
    # Simular propagação
    ep_signal = simular_propagacao(Nx, Ny, steps, am_signal, walls, source_pos, ep_pos)
    
    # Análise e visualização
    plotar_sequencia_binaria(bits, 300, modo)
    plotar_sinal_am(am_signal)
    plotar_campo_receptor(ep_signal)
    salvar_dados(ep_signal, bits, modo, ep_pos)
    exibir_estatisticas(ep_signal)
    
    plt.show(block=False)
    plt.pause(20)
    plt.close('all')
