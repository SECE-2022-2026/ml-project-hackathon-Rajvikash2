from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import uvicorn

# Initialize FastAPI app
app = FastAPI()

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

@app.post("/detect-people/")
async def detect_people(file: UploadFile = File(...)):
    """
    Endpoint to detect the number of people in an uploaded image.
    Returns the count as JSON and saves an annotated image.
    """
    try:
        # Read uploaded file
        file_contents = await file.read()
        np_image = np.frombuffer(file_contents, np.uint8)
        image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

        # Run detection
        results = model(image)

        # Draw bounding boxes and count 'person' class
        person_count = 0
        for detection in results[0].boxes:
            if int(detection.cls) == 0:  # Class ID 0 is 'person'
                person_count += 1
                x1, y1, x2, y2 = map(int, detection.xyxy[0]) 
                cv2.rectangle(image, (x1, y1), (x2, y2), (127, 255, 0), 2)  

        # Save annotated image
        output_image_path = "output.jpg"
        cv2.imwrite(output_image_path, image)

        # Return JSON response
        return JSONResponse(
            content={
                "person_count": person_count,
                "annotated_image_path": output_image_path
            }
        )

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/download-annotated-image/")
def download_annotated_image():
    """
    Endpoint to download the annotated image.
    """
    return FileResponse("output.jpg", media_type="image/jpeg", filename="output.jpg")

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)