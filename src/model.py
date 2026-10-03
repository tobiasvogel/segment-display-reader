from torch import nn

from .config import SEGMENT_TYPES, MAX_SEGMENTS


class SegmentMultiTaskCNN(nn.Module):
    """
    Shared visual backbone with two heads:
      - display-type classification: 7 / 13 / 14 / 16
      - up to 16 independent segment-state logits

    Spatial information is deliberately retained before the prediction heads.
    Segment identity depends strongly on position (e.g. B vs F, C vs E), so a
    global 1x1 average pool is unsuitable here.
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

            # Keep a coarse spatial grid instead of collapsing to 1x1.
            nn.AdaptiveAvgPool2d((6, 4)),
        )

        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 6 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
        )

        self.type_head = nn.Linear(128, len(SEGMENT_TYPES))
        self.segment_head = nn.Linear(128, MAX_SEGMENTS)

    def forward(self, x):
        features = self.shared(self.features(x))
        return {
            "type_logits": self.type_head(features),
            "segment_logits": self.segment_head(features),
        }


SegmentCNN = SegmentMultiTaskCNN
