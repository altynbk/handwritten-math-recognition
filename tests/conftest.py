import os

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")

import pytest
from PIL import Image, ImageDraw


@pytest.fixture
def symbols():
    vertical = Image.new("RGB", (200, 200), "white")
    horizontal = vertical.copy()
    ImageDraw.Draw(vertical).line((100, 30, 100, 170), fill="black", width=12)
    ImageDraw.Draw(horizontal).line((30, 100, 170, 100), fill="black", width=12)
    return vertical, horizontal
