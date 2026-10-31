import torch
from PIL import Image
import io
import torchvision.transforms as T
from models.srcnn import SRCNN
from models.espcn import ESPCN

class ModelRegistry:
    def __init__(self):
        self.models = {}
    def load(self, name, path, scale):
        if name in self.models:
            return
        if 'srcnn' in name.lower():
            m = SRCNN()
        else:
            m = ESPCN(scale=scale)
        m.load_state_dict(torch.load(path, map_location='cpu'))
        m.eval()
        self.models[name] = m

registry = ModelRegistry()
# preload demo checkpoints
try:
    from config.paths import checkpoints_dir
    registry.load('srcnn_x2', str(checkpoints_dir('srcnn_best.pt')), 2)
except Exception:
    pass
try:
    registry.load('espcn_x4', str(checkpoints_dir('espcn_best.pt')), 4)
except Exception:
    pass

def infer(model_name, image_bytes, scale):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    transform = T.Compose([T.ToTensor()])
    x = transform(img).unsqueeze(0)
    model = registry.models.get(model_name)
    if model is None:
        raise ValueError('Model not loaded')
    with torch.no_grad():
        out = model(x)
    # convert back to PIL
    out_img = T.ToPILImage()(out.squeeze(0).clamp(0,1))
    buf = io.BytesIO()
    out_img.save(buf, format='PNG')
    return buf.getvalue()
