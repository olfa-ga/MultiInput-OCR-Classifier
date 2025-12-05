import torch
import torch.nn as nn

class OCRModel(nn.Module):
    """
    OCR multi-input :
    - branche image : CNN 3 couches
    - branche type : one-hot vector
    - fusion -> classification Primary / Secondary
    """
    def __init__(self):
        super(OCRModel, self).__init__()

        # CNN pour l'image
        self.image_layer = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),        # 16x32x32
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),        # 32x16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)         # 64x8x8
        )

        image_feature_dim = 64 * 8 * 8
        type_feature_dim = 5  # one-hot vector
        total_input_dim = image_feature_dim + type_feature_dim

        self.fc = nn.Sequential(
            nn.Linear(total_input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 2)  # Primary / Secondary
        )

    def forward(self, img, type_vec):
        x = self.image_layer(img)
        x = x.view(x.size(0), -1)  # flatten
        x = torch.cat([x, type_vec], dim=1)
        return self.fc(x)
