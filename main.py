import cv2
import torch
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import time
import uvicorn
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

cred = credentials.Certificate('autopompomme-firebase-adminsdk-u7hnk-3f766622cb.json')
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://autopompomme-default-rtdb.firebaseio.com/'
})

ref = db.reference('careAi')

users_ref = ref.child('hw')
users_ref.set({
    'plantState': True
})

app = FastAPI()

# Load the custom model trained with 'yolov5s'
model_path = 'runs/train/plant_yolov5s_results/weights/best.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)  # local model

# Minimum duration for "bad leaf" detection (5 seconds)
BAD_LEAF_DURATION_THRESHOLD = 5
NO_BAD_LEAF_THRESHOLD = 1  # 1 second
DELAY_BETWEEN_DISEASE = 1


class State:
    def __init__(self):
        self.start_time_bad_leaf = None
        self.last_disease_time = None
        self.last_no_bad_leaf_time = None


state = State()


def detect_objects(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(frame_rgb, size=640)

    bad_leaf_count = 0
    for *box, conf, cls in results.xywh[0]:
        label_name = results.names[int(cls)]
        if label_name == 'bad leaf':
            bad_leaf_count += 1

    return results, bad_leaf_count


@app.get("/video_feed")
def video_feed():
    camera = cv2.VideoCapture(0)

    def gen_frames():
        while True:
            success, frame = camera.read()
            if not success:
                break
            else:
                results, bad_leaf_count = detect_objects(frame)

                if bad_leaf_count > 0:
                    if state.start_time_bad_leaf is None:
                        state.start_time_bad_leaf = time.time()
                        state.last_no_bad_leaf_time = None
                    else:
                        elapsed_time = time.time() - state.start_time_bad_leaf
                        if elapsed_time >= BAD_LEAF_DURATION_THRESHOLD:
                            if state.last_disease_time is None or (time.time() - state.last_disease_time) >= DELAY_BETWEEN_DISEASE:
                                print("disease")
                                plantState = False
                                yield "disease"
                                state.last_disease_time = time.time()
                                users_ref.set({'plantState': plantState})
                            state.start_time_bad_leaf = None
                            time.sleep(1)

                # If no "bad leaf" is detected for more than 1 second, set plantState to True
                if bad_leaf_count == 0:
                    if state.last_no_bad_leaf_time is None:
                        state.last_no_bad_leaf_time = time.time()
                    else:
                        elapsed_time_no_bad_leaf = time.time() - state.last_no_bad_leaf_time
                        if elapsed_time_no_bad_leaf >= NO_BAD_LEAF_THRESHOLD:
                            plantState = True
                            state.last_no_bad_leaf_time = None
                            users_ref.set({'plantState': plantState})

                for *box, conf, cls in results.xywh[0]:
                    label = f'{results.names[int(cls)]}: {conf:.2f}'
                    frame = cv2.rectangle(frame, (int(box[0]), int(box[1]), int(
                        box[0] + box[2]), int(box[1] + box[3])), (255, 0, 0), 3)
                    frame = cv2.putText(frame, label, (int(box[0]), int(
                        box[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3, cv2.LINE_AA)

                ret, jpeg = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' +
                       jpeg.tobytes() + b'\r\n')

    return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace;boundary=frame")


if __name__ == "__main__":
    uvicorn.run(app=app, port=8001)
