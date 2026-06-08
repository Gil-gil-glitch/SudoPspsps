import pyaudio

p = pyaudio.PyAudio()

try:
    # This automatically grabs whatever is set as the system default input
    default_input = p.get_default_input_device_info()
    print(f"Default Microphone Found!")
    print(f"Name: {default_input.get('name')}")
    print(f"Index: {default_input.get('index')}")
except Exception as e:
    print("Could not find a default input device.")
    # Fallback: List everything again just in case
    for i in range(p.get_device_count()):
        print(f"Index {i}: {p.get_device_info_by_index(i).get('name')}")

p.terminate()