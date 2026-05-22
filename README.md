# Turtle Draw — Ponderada ROS 2

Desenha os contornos de uma imagem com a tartaruga do turtlesim usando uma
pipeline de visão computacional implementada **do zero com NumPy**.

---

## Estrutura do projeto

```
turtle_draw_ws/
├── src/
│   └── turtle_draw/
│       ├── package.xml
│       ├── setup.py
│       ├── setup.cfg
│       ├── resource/turtle_draw       
│       ├── images/
│       │   └── dog.jpg                ← coloque a imagem aqui para ser processada (exemplo aqui seria o cachorro
│       └── turtle_draw/
│           ├── __init__.py
│           ├── cv_pipeline.py         ← Etapas 1 e 2: pré-processamento + bordas
│           ├── path_planner.py        ← Etapa 3: planejamento de caminho
│           └── turtle_controller.py  ← Etapa 4: nó ROS 2 principal
└── README.md
```

---

## Pipeline de Visão Computacional (resumo)

| Etapa | Método | Biblioteca |
|-------|--------|-----------|
| Resize | Nearest-neighbour com `np.linspace` + indexação fancy | NumPy |
| Grayscale | Luminância BT.601: `Y = 0.114B + 0.587G + 0.299R` | NumPy |
| Gaussian Blur | Convolução 2D com kernel Gaussiano separável via stride tricks | NumPy |
| Sobel | Kernels Kx/Ky 3×3 → magnitude `√(Gx²+Gy²)` | NumPy |
| Threshold | Histerese dupla (strong/weak) + dilatação por convolução | NumPy |

> **OpenCV** é usado **única e exclusivamente** em `cv2.imread` para carregar a imagem.

---

## Pré-requisitos

```bash
# ROS 2 Humble + turtlesim
sudo apt install ros-humble-turtlesim

# Python
pip install numpy opencv-python matplotlib
```

---

## Configuração inicial

1. Copie a imagem do cachorro para a pasta de assets:

```bash
cp /caminho/para/dog.jpg \
   turtle_draw_ws/src/turtle_draw/images/dog.jpg
```

2. Build do workspace:

```bash
cd turtle_draw_ws
colcon build --symlink-install
source install/setup.bash
```

---

## Execução

Abra **dois terminais** (ambos dentro do workspace com source feito):

### Terminal 1 — turtlesim

```bash
source /opt/ros/humble/setup.bash
ros2 run turtlesim turtlesim_node
```

### Terminal 2 — controlador

```bash
cd turtle_draw_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run turtle_draw turtle_controller
```

#### Parâmetros opcionais

```bash
ros2 run turtle_draw turtle_controller \
  --ros-args \
  -p image_path:=/caminho/absoluto/dog.jpg \
  -p max_points:=600 \
  -p sigma:=1.5 \
  -p low_ratio:=0.12 \
  -p high_ratio:=0.30 \
  -p visualize:=true
```

| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| `image_path` | `share/turtle_draw/images/dog.jpg` | Caminho absoluto para a imagem |
| `max_points` | `500` | Máximo de waypoints (afeta velocidade de desenho) |
| `jump_threshold` | `0.8` | Distância (unid. turtlesim) para levantar a caneta |
| `sigma` | `1.5` | Desvio-padrão do filtro Gaussiano |
| `ksize` | `5` | Tamanho do kernel Gaussiano |
| `low_ratio` | `0.15` | Limiar fraco (fração do gradiente máximo) |
| `high_ratio` | `0.35` | Limiar forte |
| `max_dim` | `400` | Lado máximo da imagem redimensionada (px) |
| `visualize` | `false` | Salva figura das etapas em `/tmp/cv_pipeline.png` |

---

## Justificativa das escolhas

### 1. Redimensionamento (nearest-neighbour)
Reduz a imagem para ≤ 400 px no lado maior antes de qualquer processamento.
Mantém a convolução rápida O(H·W·k²) e limita o número de pixels de borda.

### 2. Grayscale (BT.601)
Os coeficientes ITU-R BT.601 modelam a sensibilidade perceptual humana ao
verde (~59 %), vermelho (~30 %) e azul (~11 %), preservando o contraste
visualmente relevante melhor do que uma média simples dos canais.

### 3. Gaussian Blur (σ = 1.5, kernel 5×5)
O buldogue francês tem textura de pelos que gera falsos positivos no Sobel.
O Gaussiano é ótimo para isso: separável, isotrópico e de suporte limitado
(o kernel 5×5 captura ≥ 99 % da energia de σ = 1.5).

### 4. Sobel (3×3)
Aproxima ∂I/∂x e ∂I/∂y. Os pesos ±2 suavizam na direção perpendicular,
dando melhor relação sinal-ruído que uma diferença finita simples.

### 5. Histerese dupla
Dois limiares classificam pixels em *strong* (sempre borda) e *weak* (borda
só se adjacente a *strong*), produzindo contornos contínuos com menos ruído.

### 6. Nearest-Neighbour tour
Minimiza o deslocamento total da tartaruga. Complexidade O(N²) com NumPy
— viável para N ≤ 600 pontos.

---

## Dependências

```
ros-humble-turtlesim
python3-numpy
python3-opencv
python3-matplotlib
```
