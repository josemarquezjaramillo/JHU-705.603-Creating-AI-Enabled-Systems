import cv2
import numpy as np

class VideoProcessing:

    def __init__(self, udp_url, skip_every_frame=30):
        """
        Initialize the VideoProcessing class.

        Parameters:
        - udp_url: URL of the UDP stream.
        - skip_every_frame: Number of frames to skip between processing frames.
        """
        self.stream_capture = cv2.VideoCapture(udp_url, cv2.CAP_FFMPEG)
        self.skip_every_frame = skip_every_frame
        self.frame_count = 0
        self.output_size = 416  # Assuming an output size, you can change it as required

    def resize_image(self, image):
        """
        Resize the image to the specified output size.

        Parameters:
        - image: The input image to be resized.

        Returns:
        - image: The resized image.
        """
        height, width, _ = image.shape
        if height != self.output_size or width != self.output_size:
            image = cv2.resize(image, (self.output_size, self.output_size))
        return image

    def scale_image(self, image):
        """
        Scale the image by normalizing pixel values.

        Parameters:
        - image: The input image to be normalized.

        Returns:
        - image: The normalized image.
        """
        image = image.astype(np.float32) / 255.0  # Normalize pixel values to [0, 1]
        return image

    def capture_udp_stream(self):
        """
        Captures video from a UDP stream and yields processed frames.

        Yields:
        - frame: The processed frame from the UDP stream.
        """
        # Open a connection to the UDP stream
        if not self.stream_capture.isOpened():
            print(f"Error: Unable to open UDP stream")
            return

        while True:
            # Read a frame from the UDP stream
            ret, frame = self.stream_capture.read()

            if not ret:
                print("Error: Unable to read frame from UDP stream")
                break

            self.frame_count += 1
            if self.frame_count % self.skip_every_frame == 0:
                # Resize and scale the frame
                frame = self.resize_image(frame)
                frame = self.scale_image(frame)
                yield frame

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.stream_capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    udp_url = 'udp://127.0.0.1:23000'  # Replace with your UDP stream URL
    stream = VideoProcessing(udp_url, skip_every_frame=30)
    for frame in stream.capture_udp_stream():
        cv2.imshow('UDP Stream', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
