import torchvision

# Download MNIST dataset
torchvision.datasets.MNIST('dataset/', train=True, download=True)
torchvision.datasets.MNIST('dataset/', train=False, download=True)
