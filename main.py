from imageai.Detection import ObjectDetection
import os
exec_path = os.getcwd()
detection = ObjectDetection()
detection.setModelTypeAsRetinaNet()
detection.setModelPath(os.path.join(exec_path,
                                    "retinanet_resnet50_fpn_coco-eeacb38b.pth"))
detection.loadModel()

detections = (detection.detectObjectsFromImage
              (input_image=(os.path.join(exec_path, "1.jpg")),
                output_image_path=(os.path.join(exec_path, "out.jpg"))))

count = 0
for detection in detections:
    if detection["name"] == "person":
        count += 1
print(count)