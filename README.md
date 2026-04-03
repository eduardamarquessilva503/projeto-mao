# Libras-Detector

Um detector de gestos em Linguagem Brasileira de Sinais (Libras) em tempo real usando visão computacional e machine learning.

## 🎯 Objetivo

Reconhecer e classificar gestos da Libras através de uma câmera webcam em tempo real, utilizando detecção de landmarks de mão com MediaPipe e um modelo de machine learning (Random Forest).

## 🚀 Funcionalidades

- **Detecção em Tempo Real**: Captura e processa vídeo da câmera em tempo real
- **Reconhecimento de Gestos**: Classifica gestos da Libras usando modelo treinado
- **Interface Interativa**: Visualização dos landmarks da mão e predições
- **Coleta de Dados**: Sistema para coletar e armazenar exemplos de gestos
- **Treinamento Automatizado**: Pipeline de treinamento com validação

## 📋 Requisitos

- Python 3.7+
- OpenCV (`cv2`)
- MediaPipe
- scikit-learn
- pandas

## 📦 Instalação

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/libras-detector.git
cd libras-detector
```

2. Instale as dependências:
```bash
pip install opencv-python mediapipe scikit-learn pandas
```

## 🎮 Como Usar

### 1. Coletar Dados de Treinamento

Execute o script de coleta para capturar gestos:
```bash
python coletar.py
```

Siga as instruções na tela para capturar amostras de cada gesto.

### 2. Treinar o Modelo

Após coletar dados suficientes:
```bash
python treinar.py
```

Isso criará um arquivo `modelo.pkl` com o modelo treinado.

### 3. Executar o Detector

Inicie o detector em tempo real:
```bash
python main.py
```

Aponte a câmera para suas mãos e realize os gestos para serem classificados.

## 📂 Estrutura do Projeto

```
libras-detector/
├── main.py           # Aplicação principal (detector em tempo real)
├── detector.py       # Classe HandDetector (MediaPipe)
├── coletar.py        # Script de coleta de dados
├── treinar.py        # Pipeline de treinamento do modelo
├── utils.py          # Funções utilitárias
├── dados.csv         # Dados coletados (gerado)
├── modelo.pkl        # Modelo treinado (gerado)
└── README.md         # Este arquivo
```

## 🔧 Componentes

### `detector.py`
Define a classe `HandDetector` que utiliza MediaPipe para:
- Detectar landmarks das mãos (21 pontos)
- Desenhar a estrutura da mão na imagem
- Normalizar as coordenadas

### `coletar.py`
Script interativo para capturar gestos:
- Pressione a letra correspondente ao gesto
- Coleta coordenadas normalizadas dos landmarks
- Armazena em `dados.csv`

### `treinar.py`
Pipeline de treinamento:
- Carrega dados do CSV
- Treina modelo RandomForest
- Valida com conjunto de teste
- Salva modelo em `modelo.pkl`

### `main.py`
Aplicação principal:
- Carrega modelo treinado
- Processa stream da câmera
- Exibe predições em tempo real
- Interface com buttons e feedback visual

### `utils.py`
Funções auxiliares para:
- Normalização de landmarks
- Processamento de dados

## 📊 Modelo

- **Algoritmo**: Random Forest Classifier
- **Features**: Landmarks normalizados da mão (42 valores)
- **Classes**: Gestos/Letras da Libras

## 🤝 Contribuindo

Contribuições são bem-vindas! Sinta-se livre para:
- Melhorar a acurácia do modelo
- Adicionar novos gestos
- Otimizar o desempenho
- Corrigir bugs

## 📝 Licença

Este projeto está sob a licença MIT.

## 👨‍💻 Autor

Criado com ❤️ para inclusão através da tecnologia.

---

**Nota**: Este projeto foi desenvolvido para fins educacionais e é um passo inicial para sistemas mais complexos de reconhecimento de Libras.
