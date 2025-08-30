import cv2
import sys

def decode_fourcc(fourcc_int):
    """Decodes a FOURCC integer into a human-readable string."""
    if fourcc_int == 0.0:
        return "N/A"
    try:
        return "".join([chr((int(fourcc_int) >> 8 * i) & 0xFF) for i in range(4)])
    except Exception:
        return str(fourcc_int)

def check_camera_modes(camera_index=0):
    """
    Checks and lists available properties and attempts to list common modes for a given camera index.
    """
    print(f"--- Checking Camera Index: {camera_index} ---")

    # Try different backends as they can expose different capabilities
    backends = {
        "Default": None,
        "DSHOW": cv2.CAP_DSHOW,
        "MSMF": cv2.CAP_MSMF
    }

    for be_name, be_const in backends.items():
        print(f"\n>>> Trying Backend: {be_name}")
        try:
            if be_const is None:
                cap = cv2.VideoCapture(camera_index)
            else:
                cap = cv2.VideoCapture(camera_index, be_const)

            if not cap.isOpened():
                print(f"    Could not open camera with {be_name} backend.")
                if cap: cap.release()
                continue

            print("    Camera opened successfully.")
            backend_name = cap.getBackendName()
            print(f"    OpenCV Backend in use: {backend_name}")

            # Common resolutions to test
            resolutions = [
                (640, 480),
                (800, 600),
                (1280, 720),
                (1920, 1080)
            ]
            
            # Common frame rates to test
            framerates = [60, 30, 24, 15]

            # Common formats to test
            formats = ['MJPG', 'YUY2']

            supported_modes = set()

            for width, height in resolutions:
                for fps in framerates:
                    for fmt in formats:
                        # Set format first, as it's often the deciding factor for resolution/fps support
                        fourcc = cv2.VideoWriter_fourcc(*fmt)
                        cap.set(cv2.CAP_PROP_FOURCC, fourcc)

                        # Then set resolution and FPS
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                        cap.set(cv2.CAP_PROP_FPS, fps)

                        # Read back the values to see what the driver actually set
                        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        actual_fps = cap.get(cv2.CAP_PROP_FPS)
                        actual_fourcc_int = cap.get(cv2.CAP_PROP_FOURCC)
                        actual_fourcc_str = decode_fourcc(actual_fourcc_int)

                        # Check if the camera accepted our settings
                        if (actual_width == width and 
                            actual_height == height and 
                            abs(actual_fps - fps) < 1.0 and # Allow for small floating point differences
                            actual_fourcc_str == fmt):
                            mode_tuple = (actual_width, actual_height, int(round(actual_fps)), actual_fourcc_str)
                            supported_modes.add(mode_tuple)

            if supported_modes:
                print("\n    --- Discovered Supported Modes ---")
                # Sort for readability: by width, then height, then FPS
                sorted_modes = sorted(list(supported_modes), key=lambda x: (x[0], x[1], x[3], x[2]), reverse=True)
                for w, h, f, fcc in sorted_modes:
                    print(f"    -> {w}x{h} @ {f} FPS, Format: {fcc}")
                print("    ----------------------------------\n")
            else:
                print("    Could not auto-discover specific modes. The camera might not allow setting them this way.")

            cap.release()
            print(f"    Camera released.")

        except Exception as e:
            print(f"    An error occurred with {be_name} backend: {e}")
            if 'cap' in locals() and cap and cap.isOpened():
                cap.release()

if __name__ == '__main__':
    cam_idx_to_test = 1
    if len(sys.argv) > 1:
        try:
            cam_idx_to_test = int(sys.argv[1])
        except ValueError:
            print(f"Invalid camera index '{sys.argv[1]}'. Using index 0.")

    check_camera_modes(camera_index=cam_idx_to_test)

