from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
import io, torch
from PIL import Image
import torchvision.transforms as T
from models.srcnn import SRCNN
from models.espcn import ESPCN

app = FastAPI(title="Surveillance SR API")

# Simple in-memory registry
models = {}
def load_model(name, path, scale):
    if name in models: return
    if 'srcnn' in name:
        m = SRCNN()
    else:
        m = ESPCN(scale=scale)
    m.load_state_dict(torch.load(path, map_location='cpu'))
    m.eval()
    models[name] = m

# try load existing checkpoints
import os
from config.paths import checkpoints_dir
ckpt_srcnn = str(checkpoints_dir('srcnn_real_best.pt'))
ckpt_espcn = str(checkpoints_dir('espcn_real_best.pt'))
if os.path.exists(ckpt_srcnn):
    load_model('srcnn_x2', ckpt_srcnn, 2)
if os.path.exists(ckpt_espcn):
    load_model('espcn_x4', ckpt_espcn, 4)

@app.get("/api/v1/health")
def health():
    return {"status":"ok"}

@app.get("/api/v1/models")
def list_models():
    return {"models": list(models.keys())}

@app.post("/api/v1/super-resolve")
async def super_resolve(file: UploadFile = File(...), model: str = "srcnn_x2"):
    if model not in models:
        raise HTTPException(status_code=404, detail="Model not found")
    data = await file.read()
    img = Image.open(io.BytesIO(data)).convert('RGB')
    x = T.ToTensor()(img).unsqueeze(0)
    m = models[model]
    with torch.no_grad():
        out = m(x)
    out_img = T.ToPILImage()(out.squeeze(0).clamp(0,1))
    buf = io.BytesIO()
    out_img.save(buf, format='PNG')
    return StreamingResponse(io.BytesIO(buf.getvalue()), media_type="image/png")

@app.post("/api/v1/compare")
async def compare(file: UploadFile = File(...), model_a: str = "srcnn_x2", model_b: str = "espcn_x4"):
    if model_a not in models or model_b not in models:
        raise HTTPException(status_code=404, detail="Model not found")
    data = await file.read()
    img = Image.open(io.BytesIO(data)).convert('RGB')
    x = T.ToTensor()(img).unsqueeze(0)
    results = {}
    for name in [model_a, model_b]:
        m = models[name]
        with torch.no_grad():
            out = m(x)
        out_img = T.ToPILImage()(out.squeeze(0).clamp(0,1))
        buf = io.BytesIO()
        out_img.save(buf, format='PNG')
        results[name] = buf.getvalue()
    # return as base64 json
    import base64
    return {
        "model_a": model_a,
        "model_b": model_b,
        "images": {
            model_a: base64.b64encode(results[model_a]).decode(),
            model_b: base64.b64encode(results[model_b]).decode()
        }
    }
