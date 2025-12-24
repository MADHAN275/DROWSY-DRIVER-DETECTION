import mediapipe
print(f"File: {mediapipe.__file__}")
print(f"Dir: {dir(mediapipe)}")
try:
    import mediapipe.python.solutions as solutions
    print("Found mediapipe.python.solutions")
except ImportError as e:
    print(f"Could not import mediapipe.python.solutions: {e}")
