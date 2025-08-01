from ultralytics import YOLO
import os
import cv2
import uuid
import ffmpeg
import pandas as pd
import numpy as np


def avi_to_mp4(input_file, output_file):
    try:
        (
            ffmpeg
            .input(input_file)
            .output(output_file, vcodec='libx264', acodec='aac')
            .overwrite_output()
            .run()
        )
        return output_file
    except ffmpeg.Error as e:
        print(f"Hata: {e}")


def get_equipment_names(results):
    equipment_data = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = round(float(box.conf[0]), 2)
            class_name = result.names[class_id]
            equipment_data.append({
                "Class_ID": class_id,
                "Class Name": class_name,
                "Confidence": confidence
            })
    return pd.DataFrame(equipment_data)


def check_compliancy(results):

    def is_inside(box_a, box_b):
        return(
            box_b[0]>=box_a[0] and
            box_b[1]>=box_a[1] and
            box_b[2]<=box_a[2] and
            box_b[3]<=box_a[3]
        )

    persons=[]
    helmets=[]
    vests=[]
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            xyxy = box.xyxy[0].tolist()
            label = result.names[class_id]

            if label == "person":
                persons.append(xyxy)
            elif label == "helmet":
                helmets.append(xyxy)
            elif label == "vest":
                vests.append(xyxy)

    people_info = []
    no_vest=0
    no_helmet=0
    no_compliance=0
    warn_str = ""
    for p_box in persons:
        has_helmet = any(is_inside(p_box, h) for h in helmets)
        has_vest = any(is_inside(p_box, v) for v in vests)

        people_info.append({
            'Has helmet': has_helmet,
            'Has vest': has_vest,
            'Compliance': has_helmet and has_vest,
        })

        if not has_vest and not has_helmet:
            no_compliance+=1
        if not has_vest and has_helmet:
            no_vest+=1
        if has_vest and not has_helmet:
            no_helmet+=1
    if no_vest>0:
        warn_str += f"{no_vest} person has no vest!\n"
    if no_helmet>0:
        warn_str += f"{no_helmet} person has no helmet!\n"
    if no_compliance>0:
        warn_str += f"{no_compliance} person has no compliance!\n"


    return pd.DataFrame(people_info), warn_str





model = YOLO("best.pt")

def get_predict(image):
    results = model.predict(image, save=True, conf=0.4, project="./", name="results", exist_ok=True)
    table = get_equipment_names(results)
    table2, warn_str = check_compliancy(results)
    pic=[pic for pic in os.listdir("./results/")][-1]
    pic_path=os.path.join("./results/", pic)
    pic=cv2.imread(pic_path)
    window_name="output"

    return pic, table, table2, warn_str


def get_predict_from_video2(video_path):
    try:
        print(f"Video işleniyor: {video_path}")
        unique_name = f"video_results_{uuid.uuid4().hex[:8]}"

        results = model.track(
            video_path,
            save=True,
            conf=0.4,
            iou=0.5,
            tracker="bytetrack.yaml",
            project="./",
            name=unique_name,
            exist_ok=True,
            show=False
        )

        # Tracker ile elde edilen bilgiler
        data = []
        for result in results:
            for box in result.boxes:
                track_id = int(box.id[0]) if box.id is not None else -1
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = result.names[class_id]

                data.append({
                    "Track ID": track_id,
                    "Class Name": class_name,
                    "Conf": round(conf, 2)
                })

        df = pd.DataFrame(data).drop_duplicates(subset=["Track ID", "Class Name"])
        df = df.sort_values("Conf", ascending=False).drop_duplicates(subset=["Track ID"])

        _, warn_str = check_compliancy(results)

        # Video dosyasını bul
        output_dir = f"./{unique_name}/"
        video_files = [f for f in os.listdir(output_dir) if f.endswith((".mp4", ".avi"))]

        if video_files:
            yolo_output_path = os.path.join(output_dir, video_files[-1])
            video = avi_to_mp4(yolo_output_path, f'{output_dir}/video.mp4')
            return video, df, warn_str


    except Exception as e:

        print(f"Hata oluştu: {e}")

        return video_path, pd.DataFrame(), "Bir hata oluştu."



def get_predict_from_video(video_path):
    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    interval = int(cap.get(cv2.CAP_PROP_FPS))

    all_predictions=[]

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_count % interval == 0:
            results = model.predict(frame, conf=0.4)
            all_predictions.append((frame_count,results))

        frame_count += 1
    cap.release()
    print(all_predictions)

if __name__ == "__main__":
    get_predict_from_video(r"C:\Users\kadir\OneDrive\Masaüstü\#ironworker #easymoney #unionironworker.mp4")
    #get_predict("image.jpg")