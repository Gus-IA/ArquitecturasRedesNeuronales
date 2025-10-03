from sklearn.datasets import fetch_openml
import numpy as np
import torch
from torch.nn import Sequential as S
from torch.nn import Linear as L
from torch.nn import ReLU as R
import torchvision
import numpy as np
import random
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
from skimage import color
from skimage import exposure

# descargamos el dataset mnist
mnist = fetch_openml('mnist_784', version=1)
X, Y = mnist["data"].values.astype(np.float32) / 255., mnist["target"].values.astype(int)

# instanciamos la clase Dataset y pagamos los datos y los convertimos a tensores de pytorch
class Dataset(torch.utils.data.Dataset):

	# constructor
	def __init__(self, X, Y):
		self.X = torch.from_numpy(X).float()
		self.Y = torch.from_numpy(Y).long()

	# cantidad de muestras en el dataset
	def __len__(self):
		return len(self.X)

	# devolvemos el elemento `ix` del dataset
	def __getitem__(self, ix):
		return self.X[ix], self.Y[ix]

# entrenamos el modelo
def train(model, epochs = 5, batch_size=1000):
	dataset = {
		"train": Dataset(X[:60000], Y[:60000]), # 60.000 imágenes para entrenamiento
		"val": Dataset(X[60000:], Y[60000:])    # 10.000 imágenes para validación
	}
	dataloader = {
		'train': torch.utils.data.DataLoader(dataset['train'], batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True),
		'val': torch.utils.data.DataLoader(dataset['val'], batch_size=batch_size, num_workers=4, pin_memory=True)
	}
	model.cuda()
	criterion = torch.nn.CrossEntropyLoss() # función de pérdida
	optimizer = torch.optim.Adam(model.parameters()) # optimizador
	for e in range(1, epochs+1):
		print(f"epoch: {e}/{epochs}")
		# entrenamiento
		model.train()
		for batch_ix, (x, y) in enumerate(dataloader['train']):
			x, y = x.cuda(), y.cuda()
			optimizer.zero_grad()
			with torch.autocast(device_type='cuda', dtype=torch.bfloat16): # precisión mixta: ocupa menos espacio con float16
				outputs = model(x)
				loss = criterion(outputs, y)
			loss.backward()
			optimizer.step()
			if batch_ix % 10 == 0:
				loss, current = loss.item(), (batch_ix + 1) * len(x)
				print(f"loss: {loss:.4f} [{current:>5d}/{len(dataset['train']):>5d}]")
		# validación
		model.eval()
		val_loss, val_acc = [], []
		with torch.no_grad():
			for batch_ix, (x, y) in enumerate(dataloader['val']):
				x, y = x.cuda(), y.cuda()
				with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
					outputs = model(x)
					loss = criterion(outputs, y)
				val_loss.append(loss.item())
				val_acc.append((outputs.argmax(1) == y).float().mean().item())
		print(f"val_loss: {np.mean(val_loss):.4f} val_acc: {np.mean(val_acc):.4f}")
		

# ---- Perceptrón Multicapa ----


model = S(L(784,128),R(),L(128,10))
print(model)

train(model)


# ---- Redes Neuronales Convolucionales

# descargamos el dataset CIFAR10 con imagenes
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# separamos en train y test
train_imgs, train_labels = np.array([np.array(i[0]) for i in trainset]), np.array([i[1] for i in trainset])
test_imgs, test_labels = np.array([np.array(i[0]) for i in testset]), np.array([i[1] for i in testset])

train_imgs.shape, test_imgs.shape



# mostramos una imagen aleatoria
ix = random.randint(0, len(train_imgs))
img, label = train_imgs[ix], train_labels[ix]

plt.imshow(img)
plt.title(classes[label])
plt.show()





# convertimos la iamgen RGB a escala de grises
# resaltando los bordes horizontales
img = color.rgb2gray(img)

kernel = np.array([[1,1,1],
                   [0,0,0],
                   [-1,-1,-1]])

edges = scipy.signal.convolve2d(img, kernel, 'valid')
edges = exposure.equalize_adapthist(edges/np.max(np.abs(edges)), clip_limit=0.03)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
ax1.imshow(img, cmap=plt.cm.gray)
ax2.imshow(edges, cmap=plt.cm.gray)
plt.show()



# resaltado los bordes verticales, traspuesto a l'anterior ejemplo
kernel = np.array([[1,0,-1],
                   [1,0,-1],
                   [1,0,-1]])

edges = scipy.signal.convolve2d(img, kernel, 'valid')
edges = exposure.equalize_adapthist(edges/np.max(np.abs(edges)), clip_limit=0.03)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
ax1.imshow(img, cmap=plt.cm.gray)
ax2.imshow(edges, cmap=plt.cm.gray)
plt.show()



# detecta todos los bordes
kernel = np.array([[0,-1,0],
                   [-1,4,-1],
                   [0,-1,0]])

edges = scipy.signal.convolve2d(img, kernel, 'valid')
edges = exposure.equalize_adapthist(edges/np.max(np.abs(edges)), clip_limit=0.03)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
ax1.imshow(img, cmap=plt.cm.gray)
ax2.imshow(edges, cmap=plt.cm.gray)
plt.show()


# lo mismo pero usando pytorch

# mostramos y usamos una imagen aleatoria
ix = random.randint(0, len(train_imgs))
img, label = train_imgs[ix], train_labels[ix]

plt.imshow(img)
plt.title(classes[label])
plt.show()




# convertir la imágen en tensor con dimensiones (N, C_in, H, W)

img_tensor = torch.from_numpy(img / 255.).unsqueeze(0)
img_tensor = img_tensor.permute(0, 3, 1, 2).float()

img_tensor.shape, img_tensor.dtype




# aplicamos 10 filtros de tamaño 3x3

conv = torch.nn.Conv2d(in_channels = 3, out_channels = 10, kernel_size = 3)

output = conv(img_tensor)

# dimensiones: (N, #filtros, H', W')
print(output.shape)




conv = torch.nn.Conv2d(in_channels = 3, out_channels = 10, kernel_size = 3, padding = 1, stride = 1)

output = conv(img_tensor)

# dimensiones: (N, #filtros, H', W')
print(output.shape)




conv = torch.nn.Conv2d(in_channels = 3, out_channels = 10, kernel_size = 3, padding = 1, stride = 2)

output = conv(img_tensor)

# dimensiones: (N, #filtros, H', W')
print(output.shape)




# ---- Pooling ----

# mostramos y usamos una imagen aleatoria
ix = random.randint(0, len(train_imgs))
img, label = train_imgs[ix], train_labels[ix]

plt.imshow(img)
plt.title(classes[label])
plt.show()



# reducimos la imagen a la mitad
pool = torch.nn.MaxPool2d(3, padding=1, stride=2)

img_tensor = torch.from_numpy(img / 255.).unsqueeze(0).permute(0, 3, 1, 2).float()
output = pool(img_tensor)
print(output.shape)


plt.imshow(output.squeeze(0).permute(1,2,0))
plt.show()




# ---- Redes convolucionales ----

# generamos la red convolucional definiendo las capas
def block(c_in, c_out, k=3, p=1, s=1, pk=2, ps=2):
    return torch.nn.Sequential(
        torch.nn.Conv2d(c_in, c_out, k, padding=p, stride=s),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(pk, stride=ps)
    )

# función para aplicar la red convolucional 
class CNN(torch.nn.Module):
  def __init__(self, n_channels=1, n_outputs=10):
    super().__init__()
    self.conv1 = block(n_channels, 64)
    self.conv2 = block(64, 128)
    self.fc = torch.nn.Linear(128*7*7, n_outputs)

# flujo de datos entre capas
  def forward(self, x):
    print("Dimensiones:")
    print("Entrada: ", x.shape)
    x = self.conv1(x)
    print("conv1: ", x.shape)
    x = self.conv2(x)
    print("conv2: ", x.shape)
    x = x.view(x.shape[0], -1)
    print("pre fc: ", x.shape)
    x = self.fc(x)
    print("Salida: ", x.shape)
    return x

model = CNN()

# batch de 64 imágenes aleatorias de 28x28
output = model(torch.randn(64, 1, 28, 28))




class CNN(torch.nn.Module):
  def __init__(self, n_channels=1, n_outputs=10):
    super().__init__()
    self.conv1 = block(n_channels, 64)
    self.conv2 = block(64, 128)
    self.fc = torch.nn.Linear(128*7*7, n_outputs)

  def forward(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = x.view(x.shape[0], -1)
    x = self.fc(x)
    return x
  

# usamos el dataset mnist
class Dataset(torch.utils.data.Dataset):
	def __init__(self, X, Y):
		self.X = torch.tensor(X).float()
		self.Y = torch.tensor(Y).long()

	def __len__(self):
		return len(self.X)

	# devolvemos las imágenes con dimensiones (C, H, W)
	def __getitem__(self, ix):
		return self.X[ix].reshape(1, 28, 28), self.Y[ix]
	

model = CNN()

train(model)

