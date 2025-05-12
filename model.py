# This file contains the code used to make the program for the Flask-ML server. Final code is in app.py

import subprocess
import ollama
import os

# Function to extract frames from a video using ffmpeg

def extract_frames_ffmpeg(video_path, output_folder, fps=1):
    """
    Extracts frames from a video using FFmpeg.
    Parameters:
    video_path: Path to the input video.
    output_folder: Folder to save extracted frames.
    fps: Frames per second to extract (default: 1 frame per second).
    """
    os.makedirs(output_folder, exist_ok=True)
    
    output_pattern = os.path.join(output_folder, "frame_%04d.jpg")
    command = [
        "ffmpeg",
        "-i", video_path,  # Input video
        "-vf", f"fps={fps}",  # Extract at given FPS
        output_pattern  # Save frames with numbered filenames
    ]

    subprocess.run(command, check=True)
    print(f"Frames extracted to {output_folder}")


extract_frames_ffmpeg("video.mp4", "video_frames/", fps=1)

# Define the folder containing frames
folder_path = 'video_frames/'
images = sorted([os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(('.jpg', '.png', '.jpeg'))])

# Define the model to use with ollama
model = 'gemma3:4b'


# First prompt to tell the model what we are doing
init_prompt = (
    "You will receive frames from a surveillance video. Your task is to carefully analyze each frame and identify any suspicious or potentially criminal activity, "
    "such as theft, violence, trespassing, kidnapping or unusual behavior. Generate a short, descriptive sentence for each frame, focusing specifically on signs of criminal activity or threats to safety. If no criminal activity is found, just summarize whatever is happening in the frames."
)
ollama.generate(model, init_prompt)

# List to store frames
frame_summaries = []

# Process each frame individually
for idx, image in enumerate(images, start=1):
    prompt = (
    f"This is frame {idx + 1} of a video. "
    "Carefully examine the image. If there is any sign of criminal or suspicious activity, "
    "summarize the criminal action clearly in one sentence. "
    "If there is no crime, just describe what is happening in the scene in a few sentences."
)

    try:
        response = ollama.generate(model, prompt, images=[image])
        frame_summaries.append(f"Frame {idx}: {response['response']}")
    except Exception as e:
        print(f"Error processing {image}: {e}")

print("Captions:")
print(frame_summaries)

# Generate the final video summary
summary_prompt = (
    "Here are one-sentence crime-related observations from frames of a surveillance video:\n"
    + "\n".join(frame_summaries)
    + "\nBased on this, summarize the video by identifying any patterns, escalating incidents, or repeated suspicious activities. "
      "Clearly state what crimes or threats were observed and if any actions appear to be part of the same event. If no crimes are found, just summarize whatever is happening in the video" 
)
final_summary = ollama.generate(model, summary_prompt)['response']


# Print the summary
print("\nFinal Video Summary:\n", final_summary)