from pathlib import Path
from PIL import Image
from torchvision import transforms
import shutil

def load_image(path):
    img = Image.open(path).convert("RGB")
    img = img.resize((512, 512))
    img = transforms.ToTensor()(img).unsqueeze_(0)
    img = img.to("cuda")
    return img


def clear_dir(path):
    path_to_clear = Path(path)
    for f in path_to_clear.iterdir():
        if f.is_dir():
            shutil.rmtree(f)
        else:
            f.unlink()

def create_dir(path):   
    Path(path).mkdir(parents=True, exist_ok=True)
    clear_dir(path)
