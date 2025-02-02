# import zipfile
# import os
#
# # Define the paths to your zip files and the extraction directory
# zip_files = [
#     "C:/Users/Administrator/Downloads/MVet-Platform M-Vet_Hackathon24 animal_type_detection_dataset 0001.zip",
#     "C:/Users/Administrator/Downloads/MVet-Platform M-Vet_Hackathon24 animal_type_detection_dataset 0002.zip",
#     "C:/Users/Administrator/Downloads/MVet-Platform M-Vet_Hackathon24 animal_type_detection_dataset 0003.zip",
#     "C:/Users/Administrator/Downloads/MVet-Platform M-Vet_Hackathon24 animal_type_detection_dataset 0004.zip",
#     "C:/Users/Administrator/Downloads/MVet-Platform M-Vet_Hackathon24 animal_type_detection_dataset 0005.zip",
#     "C:/Users/Administrator/Downloads/MVet-Platform M-Vet_Hackathon24 animal_type_detection_dataset 0006.zip",
#     "C:/Users/Administrator/Downloads/MVet-Platform M-Vet_Hackathon24 animal_type_detection_dataset 0007.zip"
# ]
#
# extract_to = "C:/Users/Administrator/Desktop/ML/HACKATHON/TASK_ONE/TEST_IMAGES"
#
# # Create the extraction directory if it doesn't exist
# os.makedirs(extract_to, exist_ok=True)
#
# # Extract each zip file
# for zip_file in zip_files:
#     with zipfile.ZipFile(zip_file, 'r') as zip_ref:
#         zip_ref.extractall(extract_to)
#
#
import os
import csv
from ultralytics import YOLO
from glob import glob
from datetime import datetime

# Define paths
test_images_path = "C:/Users/Administrator/Desktop/ML/HACKATHON/TASK_ONE/TEST_IMAGES"  # Update with the path to your test images
output_predictions_path = "C:/Users/Administrator/Desktop/ML/HACKATHON/TASK_ONE/yolo_predictions"  # Path where predictions will be saved
model_path = "C:/Users/Administrator/Desktop/ML/HACKATHON/TASK_ONE/yolov9c.pt"  # Update with your YOLO model path

# Load the model
model = YOLO(model_path)

# Create directory for predictions
os.makedirs(output_predictions_path, exist_ok=True)

# Make predictions
results = model.predict(source=test_images_path, conf=0.25, save=True, project=output_predictions_path, name='predict', exist_ok=True)

# Prepare the submission data
data = []
for result in results:
    image_name = os.path.basename(result.path)
    class_names = result.names
    for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
        class_name = class_names[int(cls)]
        bbox = box.cpu().tolist()
        record = [image_name, class_name, conf.item()] + bbox
        data.append(record)

# Create the submission file
submission_file = f'submission_{int(datetime.now().timestamp())}.csv'
with open(submission_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    header = ['filename', 'class', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax']
    writer.writerow(header)
    writer.writerows(data)

print(f'Submission file created: {submission_file}')
