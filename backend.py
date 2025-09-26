from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io

# Khởi tạo app FastAPI
app = FastAPI()

# Load model Keras
model = load_model("modells/cnn_baseline.keras")

# Giả sử bạn có số lớp (classes) = 134 (VNOnDB)
# Bạn cần sửa danh sách này theo dataset của bạn
class_names = [str(i) for i in range(10)]

# Hàm xử lý ảnh trước khi đưa vào model
def preprocess_image(image_bytes, target_size=(32, 32)):
    img = Image.open(io.BytesIO(image_bytes)).convert("L")  # chuyển grayscale
    img = img.resize(target_size)  # resize về (32,32)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=(0, -1))  # shape: (1,32,32,1)
    return img_array

@app.post("/predict/char")
async def predict_char(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        img_array = preprocess_image(contents)
        preds = model.predict(img_array)
        pred_class = np.argmax(preds, axis=1)[0]
        return {"result": class_names[pred_class]}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/predict/line")
async def predict_line(image: UploadFile = File(...)):
    # Demo: xử lý giống predict_char (bạn có thể thay bằng CRNN sau này)
    try:
        contents = await image.read()
        img_array = preprocess_image(contents)
        preds = model.predict(img_array)
        pred_class = np.argmax(preds, axis=1)[0]
        return {"result": f"Dòng chữ (demo): {class_names[pred_class]}"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run("backend:app", host="127.0.0.1", port=8000, reload=True)
