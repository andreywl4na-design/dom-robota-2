from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

# Назва твоєї картинки
image_path = "photo.jpg"

image = cv2.imread(image_path)

if image is None:
    print("Помилка: зображення не знайдено!")
    exit()

results = model(image)

for result in results:

    boxes = result.boxes

    for box in boxes:

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        class_id = int(box.cls[0])
        label = model.names[class_id]

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(
            image,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2
        )

        print(f"{label}: ({x1}, {y1}) -> ({x2}, {y2})")

cv2.imshow("Detected Patterns", image)

cv2.waitKey(0)
cv2.destroyAllWindows()