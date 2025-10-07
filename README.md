# Deep Learning con MNIST, CIFAR-10 y Transformers

Este repositorio explora diferentes técnicas de aprendizaje profundo aplicadas a visión computacional usando PyTorch y Scikit-learn. Abarca desde modelos básicos como perceptrones hasta arquitecturas más complejas como CNNs, LeNet-5, AlexNet, VGG, ResNet y Transformers.

---

## 📚 Contenidos

El proyecto incluye:

### 1. **Clasificación de MNIST**
- Descarga y preprocesamiento del dataset MNIST.
- Entrenamiento con un perceptrón multicapa (MLP).
- Uso de redes convolucionales simples.

### 2. **Análisis y visualización de CIFAR-10**
- Descarga del dataset CIFAR-10.
- Conversión a escala de grises.
- Detección de bordes usando convoluciones manuales.
- Visualización de filtros y pooling con PyTorch.

### 3. **Redes Neuronales Convolucionales (CNNs)**
- Implementación de redes convolucionales personalizadas.
- Arquitecturas famosas: 
  - LeNet-5
  - AlexNet
  - VGG-16
  - ResNet-34

### 4. **Transformers para clasificación y captioning**
- División de imágenes en parches (patches).
- Self-attention, multi-head attention y bloques de Transformer.
- Predicción de etiquetas en texto (captioning de dígitos).
- Entrenamiento de un modelo Transformer que predice la palabra ("cero", "uno", etc.) correspondiente al dígito en la imagen.

---

## ⚙️ Instalación

pip install -r requirements.txt

🧑‍💻 Autor

Desarrollado por Gus como parte de su aprendizaje en Python e IA.
