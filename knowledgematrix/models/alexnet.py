"""
    Implementation of AlexNet
"""
from torch import nn
from torchvision.models import alexnet, AlexNet_Weights
import torch

from knowledgematrix.neural_net import NN


class AlexNet(NN):
    """
        The AlexNet model.

        Args:
            input_shape (Tuple[int]): The shape of the input to the network (e.g., (3, 224, 224) for pretrained).
            num_classes (int): The number of classes in the dataset.
            save (bool): Whether to save the activations and preactivations of the network.
            pretrained (bool): Whether to use pretrained weights (set to True for transfer learning).
            freeze_features (bool): Whether to freeze the feature extractor layers during transfer learning.
            device (str): The device to run the network on.
    """
    def __init__(
            self,
            input_shape: tuple[int],
            num_classes: int,
            save: bool=False,
            pretrained: bool=False,
            freeze_features: bool=True,
            device: str="cpu"
        ) -> None:
        super().__init__(input_shape, save, device)

        if pretrained:
            print("Using AlexNet pretrained weights", flush=True)
            if input_shape[0] != 3:
                raise ValueError("AlexNet requires images with 3 channels. Please use input_shape=(3, -, -).")
            # Load pretrained model (ignores num_classes check for transfer learning)

            path_w = 'experiments/alexnet_imagenet/weights/pretrained-weights.pth'
            pretrained_model = alexnet()
            state_dict = torch.load(path_w, map_location=self.device)

            pretrained_model.to(self.device)
            pretrained_model.load_state_dict(state_dict)

            for layer in pretrained_model.children():
                if isinstance(layer, nn.Sequential):
                    for sublayer in layer.children():
                        if isinstance(sublayer, (nn.Conv2d, nn.Linear)):
                            self.layers.append(sublayer)
                        elif isinstance(sublayer, nn.ReLU):
                            self.relu()
                        elif isinstance(sublayer, nn.MaxPool2d):
                            self.maxpool(kernel_size=sublayer.kernel_size, stride=sublayer.stride, padding=sublayer.padding)
                        elif isinstance(sublayer, nn.Dropout):
                            self.dropout(p=sublayer.p)
                elif isinstance(layer, nn.AdaptiveAvgPool2d):
                    self.adaptiveavgpool(output_size=layer.output_size)
                    self.flatten()

            # Replace the last linear layer for the new number of classes
            if num_classes != 1000:
                # Find the last linear layer (classifier's output layer)
                for i in range(len(self.layers) - 1, -1, -1):
                    if isinstance(self.layers[i], nn.Linear) and self.layers[i].out_features == 1000:
                        in_features = self.layers[i].in_features  # Should be 4096
                        self.layers[i] = nn.Linear(in_features, num_classes)
                        break
                else:
                    raise ValueError("Could not find the output linear layer to replace for transfer learning.")

            # Optionally freeze the feature extractor (convolutional layers)
            if freeze_features:
                for layer in self.layers:
                    if isinstance(layer, nn.Conv2d):
                        for param in layer.parameters():
                            param.requires_grad = False

        else:
            self.conv(self.input_shape[0], 64, kernel_size=11, stride=4, padding=2)
            self.relu()
            self.maxpool(kernel_size=3, stride=2)
            self.conv(64, 192, kernel_size=5, padding=2)
            self.relu()
            self.maxpool(kernel_size=3, stride=2)
            self.conv(192, 384, kernel_size=3, padding=1)
            self.relu()
            self.conv(384, 256, kernel_size=3, padding=1)
            self.relu()
            self.conv(256, 256, kernel_size=3, padding=1)
            self.relu()
            self.maxpool(kernel_size=3, stride=2)

            self.adaptiveavgpool((6,6))

            self.flatten()

            self.dropout(0.5)
            self.linear(in_features=256*6*6, out_features=4096)
            self.relu()
            self.dropout(0.5)
            self.linear(in_features=4096, out_features=4096)
            self.relu()
            self.linear(in_features=4096, out_features=num_classes)