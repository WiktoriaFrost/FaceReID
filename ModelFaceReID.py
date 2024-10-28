import torch.nn as nn
import torch.nn.functional as F


class FaceEmbeddingCNN(nn.Module):
    def __init__(self, embedding_size=128):
        super(FaceEmbeddingCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # reducing image size

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)

        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(256, embedding_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))

        x = F.relu(self.conv3(x))
        x = self.pool(F.relu(self.conv4(x)))

        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten

        embedding = self.fc1(x)

        # L2 NORMALIZATION
        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding