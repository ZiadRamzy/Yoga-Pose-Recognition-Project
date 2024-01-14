import torch
import sys, os
import subprocess
from torchvision import transforms
import cv2
from time import sleep
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")


def fetch_movenet_model(download):
    if download:
        # CHECK THE OS
        try:
            subprocess.run(
                [
                    "powershell",
                    "Invoke-WebRequest",
                    "-Uri",
                    "https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/tflite/float16/4?lite-format=tflite",
                    "-OutFile",
                    "movenet_thunder.tflite",
                ],
                check=True,
            )
        except:
            pass
        try:
            subprocess.run(
                [
                    "wget",
                    "-q",
                    "-O",
                    "movenet_thunder.tflite",
                    "https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/tflite/float16/4?lite-format=tflite",
                ],
                check=True,
            )
        except:
            pass

        subprocess.run(["git", "clone", "https://github.com/tensorflow/examples.git"])
    pose_sample_rpi_path = os.path.join(
        os.getcwd(), "examples/lite/examples/pose_estimation/raspberry_pi"
    )
    sys.path.append(pose_sample_rpi_path)
    from ml import Movenet

    movenet = Movenet("movenet_thunder")
    return movenet


def calculate_angles(df):
    # create output dataframe
    df2 = pd.DataFrame()
    # iterate through columns
    for col_name in df.columns:
        # take the x column first
        if col_name.endswith("x"):
            # calculate the difference between x coordinates
            dx = df[col_name] - df["NOSE_x"]
            # repeat for y coordinates
            dy = df[col_name[:-1] + "y"] - df["NOSE_y"]
            # calculate angle from differences
            col_basename = col_name[:-1]

            df2[col_basename + "angle"] = np.arctan2(dy, dx)

    return df2


def movenet_detect(movenet, input_tensor, inference_count=1):
    image_height, image_width, channel = input_tensor.shape

    # Detect pose using the full input image
    movenet.detect(input_tensor.numpy(), reset_crop_region=True)

    # Repeatedly using previous detection result to identify the region of
    # interest and only croping that region to improve detection accuracy
    for _ in range(inference_count):
        person = movenet.detect(input_tensor.numpy(), reset_crop_region=False)
    return person


import math


def convert_to_degrees(angle):
    # Converts radians to degrees
    angle_deg = math.degrees(angle)
    if angle_deg < 0:
        angle_deg = 360 + angle_deg

    return angle_deg


def inference_pipeline(
    input_image,
    image_transform,
    model_classifier,
    movenet,
    angles_df,
    angle_threshold=0.5,
):
    input_tensor = image_transform(input_image)
    if input_tensor.shape[0] == 3:
        input_tensor = input_tensor.permute(1, 2, 0)

    person = movenet_detect(movenet, input_tensor)
    kps = person.keypoints

    # create a tensor from keypoints
    col_names = []
    kps_coords = []
    for kp in kps:
        x = kp.coordinate[0]
        y = kp.coordinate[1]
        score = kp.score
        kps_coords.extend([x, y, score])
        name = str(kp.body_part.name)
        col_names.extend([name + "_x", name + "_y", name + "_score"])

    kp_df = pd.DataFrame([kps_coords], columns=col_names)
    real_angles_df = calculate_angles(kp_df)

    kps_tensor = torch.tensor(kps_coords, dtype=torch.float32)
    kps_tensor = kps_tensor.unsqueeze(0)
    output_logs = model_classifier(kps_tensor)
    _, predicted = torch.max(output_logs, 1)
    predicted = predicted.item()

    filtered_df = angles_df[angles_df["class_no"] == predicted].drop(
        ["class_no", "class_name"], axis=1
    )

    most_problematic = np.argmax(filtered_df.values - real_angles_df.values)

    max_value = np.abs(np.max(filtered_df.values - real_angles_df.values))

    if max_value > angle_threshold:
        print("Pose:", angles_df.iloc[predicted]["class_name"])
        print(real_angles_df.columns[most_problematic].strip("_angle"))
        print("Angle offset:", convert_to_degrees(max_value))

    return most_problematic


def real_time_inference(image_transform, model_classifier, movenet):
    print("Starting real time inference")
    rep_df = pd.read_csv("correction_angles.csv")
    cap = cv2.VideoCapture(0)
    counter = 0
    while True:
        counter += 1
        ret, frame = cap.read()
        if not ret:
            print("Unable to capture video")
            break
        if counter > 60:
            logits = inference_pipeline(
                frame, image_transform, model_classifier, movenet, rep_df
            )
            counter = 0
        cv2.imshow("Camera Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Ending real time inference")


if __name__ == "__main__":
    model_classifier = torch.jit.load("model.pth")
    model_classifier.eval()
    movenet = fetch_movenet_model(False)

    image_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    real_time_inference(image_transform, model_classifier, movenet)
