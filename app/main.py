import os
import uuid
import torch
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image

from predictor import TumorPredictor
from report import generate_report_pdf

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(__file__)
MODEL_PATH  = os.path.join(BASE_DIR, "..", "ml_model", "brain_tumor_efficientnet.pth")
UPLOAD_DIR  = os.path.join(BASE_DIR, "static", "uploads")
ALLOWED_EXT = {"png", "jpg", "jpeg", "bmp", "tiff", "gif"}

os.makedirs(UPLOAD_DIR, exist_ok=True)
torch.set_num_threads(1)

# ── Startup: load model once ──────────────────────────────────────────────────
predictor: TumorPredictor | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    try:
        predictor = TumorPredictor.get_instance(MODEL_PATH)
        print("Model loaded ✓")
    except FileNotFoundError as e:
        print(f"WARNING: {e}")
    yield

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Brain Tumor Detection", lifespan=lifespan)

app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    if predictor is None:
        return JSONResponse(
            {"error": "Model not loaded. Train the model with the Jupyter notebook first."},
            status_code=503,
        )

    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ALLOWED_EXT:
        return JSONResponse({"error": "Invalid file type. Upload a JPG or PNG."}, status_code=400)

    base_name = uuid.uuid4().hex
    filename  = f"{base_name}.{ext}"
    filepath  = os.path.join(UPLOAD_DIR, filename)

    contents = await file.read()
    with open(filepath, "wb") as f:
        f.write(contents)

    try:
        image  = Image.open(filepath)
        result = predictor.predict(image, save_dir=UPLOAD_DIR, base_name=base_name)

        base_url = str(request.base_url).rstrip("/")
        result["base_name"]   = base_name
        result["orig_ext"]    = ext
        result["image_url"]   = f"{base_url}/static/uploads/{filename}"
        result["heatmap_url"] = f"{base_url}/static/uploads/{result.pop('heatmap_name')}"
        result["overlay_url"] = f"{base_url}/static/uploads/{result.pop('overlay_name')}"

        return JSONResponse(result)

    except Exception as e:
        return JSONResponse({"error": f"Prediction failed: {str(e)}"}, status_code=500)


@app.post("/report")
async def download_report(request: Request):
    data      = await request.json()
    base_name = data.get("base_name", "")
    orig_ext  = data.get("orig_ext", "")

    orig_path    = os.path.join(UPLOAD_DIR, f"{base_name}.{orig_ext}")
    overlay_path = os.path.join(UPLOAD_DIR, f"{base_name}_overlay.png")

    try:
        pdf_bytes = generate_report_pdf(data, orig_path, overlay_path)
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": 'attachment; filename="brain_tumor_report.pdf"'},
        )
    except Exception as e:
        return JSONResponse({"error": f"PDF generation failed: {str(e)}"}, status_code=500)


@app.get("/health")
async def health():
    return {"status": "ok" if predictor else "model_missing"}
