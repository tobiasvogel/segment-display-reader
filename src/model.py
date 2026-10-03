from torch import nn

from .config import SEGMENT_TYPES, MAX_SEGMENTS


class SegmentMultiTaskCNN(nn.Module):
    """
    Shared visual backbone with two heads:
      - display-type classification: 7 / 13 / 14 / 16
      - up to 16 independent segment-state logits
    """

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.20),
        )
        self.type_head = nn.Linear(96, len(SEGMENT_TYPES))
        self.segment_head = nn.Linear(96, MAX_SEGMENTS)

    def forward(self, x):
        features = self.shared(self.features(x))
        return {
            "type_logits": self.type_head(features),
            "segment_logits": self.segment_head(features),
        }


# Backward-friendly import name for callers; old 7-output checkpoints are not
# shape-compatible and should be retrained after this architecture change.
SegmentCNN = SegmentMultiTaskCNN
