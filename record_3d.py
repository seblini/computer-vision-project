import cv2
from datetime import datetime

width, height = 640, 480
fps = 20.0
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

timestamp = datetime.now().strftime("%Y-%m-%d-_%H-%M-%S")
out_path1 = f'raw/cam1_raw_{timestamp}.mp4'
out_path2 = f'raw/cam2_raw_{timestamp}.mp4'

cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)
cap1.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap2.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

out1 = cv2.VideoWriter(out_path1, fourcc, fps, (width, height))
out2 = cv2.VideoWriter(out_path2, fourcc, fps, (width, height))

print(f"Recording to: {out_path1}, {out_path2}")

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        print("Capture failed.")
        break

    out1.write(frame1)
    out2.write(frame2)

    combined = cv2.hconcat([frame1, frame2])
    cv2.imshow("Recording", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap1.release()
cap2.release()
out1.release()
out2.release()
cv2.destroyAllWindows()
