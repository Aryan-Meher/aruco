
import cv2
import cv2.aruco as aruco
import numpy as np

# Open webcam
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# ArUco dictionary
dictionary = aruco.getPredefinedDictionary(
    aruco.DICT_4X4_50
)

parameters = aruco.DetectorParameters()
parameters.detectInvertedMarker = True

detector = aruco.ArucoDetector(
    dictionary, parameters
)

print("ArUco Tracking System Started!")

while True:

    ret, frame = cap.read()

    if not ret:
        print("Camera error!")
        break

    # Detect markers
    corners, ids, rejected = detector.detectMarkers(frame)

    count = 0

    if ids is not None:

        count = len(ids)

        for i, marker_id in enumerate(ids.flatten()):

            # Get four corners
            points = corners[i][0].astype(int)

            # Draw tracking boundary
            cv2.polylines(
                frame,
                [points],
                True,
                (0, 255, 0),
                4
            )

            # Calculate center
            center = np.mean(points, axis=0).astype(int)

            # Draw center point
            cv2.circle(
                frame,
                tuple(center),
                7,
                (0, 0, 255),
                -1
            )

            # Display marker ID
            cv2.putText(
                frame,
                f"MARKER ID: {marker_id}",
                (points[0][0], max(30, points[0][1] - 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

    # Display detection status
    cv2.putText(
        frame,
        f"MARKERS DETECTED: {count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.imshow("ARUCO TRACKING SYSTEM", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
