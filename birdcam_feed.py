import cv2

def display_rtsp_feed(rtsp_url):
    """
    Reads an RTSP camera feed and displays it using OpenCV.

    Args:
        rtsp_url (str): The RTSP URL of the camera feed.
                        Example: 'rtsp://username:password@ip_address:port/path'
    """
    print(f"Attempting to open RTSP stream from: {rtsp_url}")

    # Create a VideoCapture object
    cap = cv2.VideoCapture(rtsp_url)
    # Check if camera opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video stream from {rtsp_url}")
        print("Please check the RTSP URL, network connectivity, and camera status.")
        return
    print("RTSP stream opened successfully. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        # If frame is read correctly, ret is True
        if not ret:
            print("Failed to grab frame. Exiting...")
            break

        # Display the resulting frame
        cv2.imshow('RTSP Camera Feed', frame)

        # Press 'q' on the keyboard to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the video capture object
    cap.release()
    # Destroy all the windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    rtsp_camera_url = 'rtsp://user:password@192.168.12.167:554/stream1' # Example Placeholder

    display_rtsp_feed(rtsp_camera_url)