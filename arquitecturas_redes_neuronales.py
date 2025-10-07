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
import math
import torch.nn as nn

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




# ---- Arquitecturas ----


def block(c_in, c_out, k=3, p=1, s=1):
    return torch.nn.Sequential(
        torch.nn.Conv2d(c_in, c_out, k, padding=p, stride=s),
        torch.nn.Tanh(),
        torch.nn.AvgPool2d(2, stride=2)
    )

def block2(c_in, c_out):
    return torch.nn.Sequential(
        torch.nn.Linear(c_in, c_out),
        torch.nn.ReLU()
    )

class LeNet5(torch.nn.Module):
  def __init__(self, n_channels=1, n_outputs=10):
    super().__init__()
    #self.pad = torch.nn.ConstantPad2d(2, 0.)
    self.conv1 = block(n_channels, 6, 5, p=0)
    self.conv2 = block(6, 16, 5, p=0)
    self.conv3 = torch.nn.Sequential(
        torch.nn.Conv2d(16, 120, 5, padding=0),
        torch.nn.Tanh()
    )
    self.fc1 = block2(120, 84)
    self.fc2 = torch.nn.Linear(84, 10)

  def forward(self, x):
    #x = self.pad(x)
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = x.view(x.shape[0], -1)
    x = self.fc1(x)
    x = self.fc2(x)
    return x
  


lenet5 = LeNet5()
output = lenet5(torch.randn(64, 1, 32, 32))
print(output.shape)




alexnet = torchvision.models.AlexNet()
print(alexnet)




output = alexnet(torch.randn(64, 3, 224, 224))
print(output.shape)





# existen dos variantes: vgg16 y vgg19, con 16 y 19 capas respectivamente
vgg16 = torchvision.models.vgg16()
print(vgg16)



output = vgg16(torch.randn(64, 3, 224, 224))
print(output.shape)



# existen múltiples variantes: resnet18, resnet35, resnet50, resnet101, resnet152
resnet34 = torchvision.models.resnet34()
print(resnet34)



# ---- Transformers ----

# descargamos el dataset mnist
mnist = fetch_openml('mnist_784', version=1)
X, Y = mnist["data"].values.astype(np.float32), mnist["target"].values.astype(int)

# creamos un dataset que recibe las imaǵenes y etiquetas
class Dataset(torch.utils.data.Dataset):
  def __init__(self, X, y, patch_size=(7, 7)):
    self.X = X
    self.y = y
    self.patch_size = patch_size

  def __len__(self):
    return len(self.X)

# creamos patches
  def __getitem__(self, ix):
    image = torch.from_numpy(self.X[ix]).float().view(28, 28) # 28 x 28
    h, w = self.patch_size
    patches = image.unfold(0, h, h).unfold(1, w, w) # 4 x 4 x 7 x 7
    patches = patches.contiguous().view(-1, h*w) # 16 x 49
    return patches, torch.tensor(self.y[ix]).long()
  



attn_dm = Dataset(X, Y)
imgs, labels = attn_dm[0]
print(imgs.shape, labels.shape)

# ejemplo de visualización
fig = plt.figure(figsize=(5,5))
for i in range(4):
    for j in range(4):
        ax = plt.subplot(4, 4, i*4 + j + 1)
        ax.imshow(imgs[i*4 + j].view(7, 7), cmap="gray")
        ax.axis('off')
plt.show()

# ejemplo de self attention
class ScaledDotSelfAttention(torch.nn.Module):

    def __init__(self, n_embd):
        super().__init__()

        # key, query, value projections
        self.key = torch.nn.Linear(n_embd, n_embd)
        self.query = torch.nn.Linear(n_embd, n_embd)
        self.value = torch.nn.Linear(n_embd, n_embd)

    def forward(self, x):
        B, L, F = x.size()

        # calculate query, key, values
        k = self.key(x) # (B, L, F)
        q = self.query(x) # (B, L, F)
        v = self.value(x) # (B, L, F)

        # attention (B, L, F) x (B, F, L) -> (B, L, L)
        att = (q @ k.transpose(1, 2)) * (1.0 / math.sqrt(k.size(-1)))
        att = torch.nn.functional.softmax(att, dim=-1)
        y = att @ v # (B, L, L) x (B, L, F) -> (B, L, F)

        return y

class Model(torch.nn.Module):

    def __init__(self, n_embd=7*7, seq_len=4*4):
        super().__init__()
        self.attn = ScaledDotSelfAttention(n_embd)
        self.actn = torch.nn.ReLU(inplace=True)
        self.fc = torch.nn.Linear(n_embd*seq_len, 10)

    def forward(self, x):
        x = self.attn(x)
        y = self.fc(self.actn(x.view(x.size(0), -1)))
        return y
    


model = Model()

train(model)


# ejemplo de multi head attention
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, n_embd, n_heads):
        super().__init__()
        self.n_heads = n_heads

        # key, query, value projections
        self.key = torch.nn.Linear(n_embd, n_embd*n_heads)
        self.query = torch.nn.Linear(n_embd, n_embd*n_heads)
        self.value = torch.nn.Linear(n_embd, n_embd*n_heads)

        # output projection
        self.proj = torch.nn.Linear(n_embd*n_heads, n_embd)

    def forward(self, x):
        B, L, F = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, L, F, self.n_heads).transpose(1, 3) # (B, nh, L, F)
        q = self.query(x).view(B, L, F, self.n_heads).transpose(1, 3) # (B, nh, L, F)
        v = self.value(x).view(B, L, F, self.n_heads).transpose(1, 3) # (B, nh, L, F)

        # attention (B, nh, L, F) x (B, nh, F, L) -> (B, nh, L, L)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = torch.nn.functional.softmax(att, dim=-1)
        y = att @ v # (B, nh, L, L) x (B, nh, L, F) -> (B, nh, L, F)
        y = y.transpose(1, 2).contiguous().view(B, L, F*self.n_heads) # re-assemble all head outputs side by side

        return self.proj(y)

class Model(torch.nn.Module):

    def __init__(self, n_embd=7*7, seq_len=4*4, n_heads=4*4):
        super().__init__()
        self.attn = MultiHeadAttention(n_embd, n_heads)
        self.actn = torch.nn.ReLU(inplace=True)
        self.fc = torch.nn.Linear(n_embd*seq_len, 10)

    def forward(self, x):
        x = self.attn(x)
        y = self.fc(self.actn(x.view(x.size(0), -1)))
        return y

model = Model()

train(model)


# ejemplo de encoder
class MultiHeadAttention(torch.nn.Module):

    def __init__(self, n_embd, n_heads):
        super().__init__()
        self.n_heads = n_heads

        # key, query, value projections
        self.key = torch.nn.Linear(n_embd, n_embd*n_heads)
        self.query = torch.nn.Linear(n_embd, n_embd*n_heads)
        self.value = torch.nn.Linear(n_embd, n_embd*n_heads)

        # output projection
        self.proj = torch.nn.Linear(n_embd*n_heads, n_embd)

    def forward(self, x):
        B, L, F = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, L, F, self.n_heads).transpose(1, 3) # (B, nh, L, F)
        q = self.query(x).view(B, L, F, self.n_heads).transpose(1, 3) # (B, nh, L, F)
        v = self.value(x).view(B, L, F, self.n_heads).transpose(1, 3) # (B, nh, L, F)

        # attention (B, nh, L, F) x (B, nh, F, L) -> (B, nh, L, L)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = torch.nn.functional.softmax(att, dim=-1)
        y = att @ v # (B, nh, L, L) x (B, nh, L, F) -> (B, nh, L, F)
        y = y.transpose(1, 2).contiguous().view(B, L, F*self.n_heads) # re-assemble all head outputs side by side

        return self.proj(y)

class TransformerBlock(torch.nn.Module):
    def __init__(self, n_embd, n_heads):
        super().__init__()
        self.ln1 = torch.nn.LayerNorm(n_embd)
        self.ln2 = torch.nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_heads)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(n_embd, 4 * n_embd),
            torch.nn.ReLU(),
            torch.nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        x = self.ln1(x + self.attn(x))
        x = self.ln2(x + self.mlp(x))
        return x

class Model(torch.nn.Module):

    def __init__(self, n_input=7*7, n_embd=7*7, seq_len=4*4, n_heads=4*4, n_layers=1):
        super().__init__()
        self.pos_emb = torch.nn.Parameter(torch.zeros(1, seq_len, n_embd))
        self.inp_emb = torch.nn.Linear(n_input, n_embd)
        self.tranformer = torch.nn.Sequential(*[TransformerBlock(n_embd, n_heads) for _ in range(n_layers)])
        self.fc = torch.nn.Linear(n_embd*seq_len, 10)

    def forward(self, x):
        # embedding
        e = self.inp_emb(x) + self.pos_emb
        # transformer blocks
        x = self.tranformer(e)
        # classifier
        y = self.fc(x.view(x.size(0), -1))
        return y
    


model = Model()

train(model)


# ---- Transformers II -----

# descargamos el dataset mnist
mnist = fetch_openml('mnist_784', version=1)
X, Y = mnist["data"].values.astype(np.float32), mnist["target"].values.astype(int)

# variable de las letras
vocab = 'abcdefghijklmnopqrstuvwxyz'
len_vocab = len(vocab) + 3

# función para mostrar la letra
def number2caption(ix):
    if ix == 0: return 'cero'
    if ix == 1: return 'uno'
    if ix == 2: return 'dos'
    if ix == 3: return 'tres'
    if ix == 4: return 'cuatro'
    if ix == 5: return 'cinco'
    if ix == 6: return 'seis'
    if ix == 7: return 'siete'
    if ix == 8: return 'ocho'
    if ix == 9: return 'nueve'

def caption2ixs(caption):
    return [vocab.index(c) + 3 for c in caption]

def ixs2caption(ixs):
    return ('').join([vocab[ix - 3] for ix in ixs if ix not in [0, 1, 2]])




captions = [number2caption(ix) for ix in Y]

# cada letra tiene su número (índice en el vocab)
encoded = [[1] + caption2ixs(caption) + [2] for caption in captions] 

print(captions[:3], encoded[:3])

# ejemplo de lo que se quiere conseguir
r, c = 4, 4
fig = plt.figure(figsize=(c*2, r*2))
for _r in range(r):
    for _c in range(c):
        ix = _r*c + _c
        ax = plt.subplot(r, c, ix + 1)
        img, caption = X[ix], captions[ix]
        ax.axis("off")
        ax.imshow(img.reshape(28,28), cmap="gray")
        ax.set_title(caption)
plt.show()


# creamos un dataset con las imágenes
class Dataset(torch.utils.data.Dataset):
  def __init__(self, X, y, patch_size=(7, 7)):
    self.X = X
    self.y = y

  def __len__(self):
    return len(self.X)

  def __getitem__(self, ix):
    image = torch.from_numpy(self.X[ix]).float().view(1, 28, 28) 
    return image, torch.tensor(self.y[ix]).long()
  
# lógica de los patches con una capa convolucional
class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, E, P, P)
        x = x.flatten(2)  # (B, E, N)
        x = x.transpose(1, 2)  # (B, N, E)
        return x

# creamos el modelo
class Model(torch.nn.Module):

    def __init__(self,
                 len_vocab,
                 img_size=28,
                 patch_size=7,
                 in_chans=1,
                 embed_dim=100,
                 max_len=8,
                 nhead=2,
                 num_encoder_layers=3,
                 num_decoder_layers=3,
                 dim_feedforward=400,
                 dropout=0.1
                ):
        super().__init__()

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.n_patches, embed_dim))

        self.trg_emb = nn.Embedding(len_vocab, embed_dim)
        self.trg_pos_emb = nn.Embedding(max_len, embed_dim)
        self.max_len = max_len

        self.transformer = torch.nn.Transformer(
            embed_dim, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout
        )

        self.l = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, len_vocab)

    def forward(self, images, captions):
        # embed images
        embed_imgs = self.patch_embed(images)
        embed_imgs = embed_imgs + self.pos_embed  # (B, N, E)
        # embed captions
        B, trg_seq_len = captions.shape
        trg_positions = (torch.arange(0, trg_seq_len).expand(B, trg_seq_len).to(images.device))
        embed_trg = self.trg_emb(captions) + self.trg_pos_emb(trg_positions)
        trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_len).to(images.device)
        tgt_padding_mask = captions == 0
        # transformer
        y = self.transformer(
            embed_imgs.permute(1,0,2),  # S, B, E
            embed_trg.permute(1,0,2),  # T, B, E
            tgt_mask=trg_mask, # T, T
            tgt_key_padding_mask = tgt_padding_mask
        ).permute(1,0,2) # B, T, E
        # head
        return self.fc(self.l(y))

# evaluación del modelo
    def predict(self, image, device):
        self.eval()
        self.to(device)
        with torch.no_grad():
            image = image.to(device)
            B = 1
            # start of sentence
            eos = torch.tensor([1], dtype=torch.long, device=device).expand(B, 1)
            trg_input = eos
            for _ in range(self.max_len):
                preds = self(image.unsqueeze(0), trg_input)
                preds = torch.argmax(preds, axis=2)
                trg_input = torch.cat([eos, preds], 1)
            return preds




MAX_LEN = 8

# función del batch
def collate_fn(batch):
    images, captions = zip(*batch)
    images = torch.stack(images)
    captions = [torch.nn.functional.pad(caption, (0, MAX_LEN - len(caption)), value=0) for caption in captions]
    captions = torch.stack(captions)
    return images, captions


# entrenamos el modelo
def train(model, epochs=5, batch_size=1000):
	dataset = {
		"train": Dataset(X[:60000], encoded[:60000]), # 60.000 imágenes para entrenamiento
		"val": Dataset(X[60000:], encoded[60000:])    # 10.000 imágenes para validación
	}
	dataloader = {
		'train': torch.utils.data.DataLoader(dataset['train'], batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_fn),
		'val': torch.utils.data.DataLoader(dataset['val'], batch_size=batch_size, num_workers=4, pin_memory=True, collate_fn=collate_fn)
	}
	model.cuda()
	criterion = torch.nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters())
	for e in range(1, epochs+1):
		print(f"epoch: {e}/{epochs}")
		# entrenamiento
		model.train()
		for batch_ix, (x, y) in enumerate(dataloader['train']):
			x, y = x.cuda(), y.cuda()
			optimizer.zero_grad()
			outputs = model(x, y[:,:-1])
			loss = criterion(outputs.permute(0,2,1), y[:,1:])
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
				outputs = model(x, y[:,:-1])
				loss = criterion(outputs.permute(0,2,1), y[:,1:])
				val_loss.append(loss.item())
				acc = (torch.argmax(outputs, axis=2) == y[:,1:]).sum().item() / (y[:,1:].shape[0]*y[:,1:].shape[1])
				val_acc.append(acc)
		print(f"val_loss: {np.mean(val_loss):.4f} val_acc: {np.mean(val_acc):.4f}")




model = Model(len(vocab) + 3, max_len=MAX_LEN)

train(model)


# visualización de los resultados
r, c = 5,5
fig = plt.figure(figsize=(c*2, r*2))
for _r in range(r):
    for _c in range(c):
        # ix = _r*c + _c
        ix = random.randint(0, len(X))
        ax = plt.subplot(r, c, _r*c + _c + 1)
        img, caption = torch.from_numpy(X[ix]).float().view(1, 28, 28), captions[ix]
        ax.axis("off")
        ax.imshow(img.reshape(28,28), cmap="gray")
        pred = model.predict(img, device='cuda')
        pred = ixs2caption(pred.squeeze().tolist())
        ax.set_title(f'{caption}/{pred}', color="green" if caption == pred else 'red')
plt.show()