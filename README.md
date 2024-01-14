# Yoga-Pose-Recognition-Project

Project developed at ITÜ for Deep Learning course (BLG 561E).
The project aims to detect a person doing a pose, classify it to a yoga pose from those which are available in the dataset and provide feedback to the user on correction of the pose.

The project uses MoveNet model to extract body keypoints which are then used in the fully connected network to classify the output. During training reference angles are calculated and while using the model, the user's keypoints are compared to the reference angles.

# Installation

To run the project locally, after pulling this repository it is necessary to download datasets from https://drive.google.com/drive/folders/1vPfavKfX__cgss9d1TCio53Ndx1q7fEH and decompress them to put them in the base directory. One of the datasets contains Yoga Posture Dataset (https://www.kaggle.com/datasets/tr1gg3rtrash/yoga-posture-dataset/data) from Kaggle.

Next, run the requirements with
pip install -r requirements.txt
??????????


To train and test the model run notebook.ipynb

To use the model run inference.py