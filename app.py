import os
import subprocess
from typing import TypedDict
from flask_ml.flask_ml_server import MLServer
from flask_ml.flask_ml_server.models import (
    ParameterSchema,
    TaskSchema,
    ResponseBody,
    TextResponse,
    BatchTextResponse,
)

import ollama

server = MLServer(__name__)

VIDEO_PATH = "video.mp4"
FRAME_FOLDER = "video_frames/"
MODEL_NAME = "gemma3:4b"


class Inputs(TypedDict):
    pass

class Parameters(TypedDict):
    fps: int  

from flask_ml.flask_ml_server.models import IntParameterDescriptor

def create_video_summary_schema() -> TaskSchema:
    fps_param_schema = ParameterSchema(
        key="fps",
        label="Frame Rate (fps)",
        subtitle="Set how many frames per second to extract from the video",
        value=IntParameterDescriptor(default=1, min=1, max=30),
    )

    return TaskSchema(inputs=[], parameters=[fps_param_schema])

def extract_frames_ffmpeg(video_path, output_folder, fps=1):
    os.makedirs(output_folder, exist_ok=True)
    output_pattern = os.path.join(output_folder, "frame_%04d.jpg")
    command = [
        "ffmpeg",
        "-i", video_path,
        "-vf", f"fps={fps}",
        output_pattern
    ]
    subprocess.run(command, check=True)

@server.route(
    "/summarize",
    task_schema_func=create_video_summary_schema,
    short_title="Video Summarization",
    order=0
)
def summarize_video(inputs: Inputs, parameters: Parameters):  

    fps = parameters.get("fps", 1)

   
    extract_frames_ffmpeg(VIDEO_PATH, FRAME_FOLDER, fps=fps)

    
    images = sorted([
        os.path.join(FRAME_FOLDER, f)
        for f in os.listdir(FRAME_FOLDER)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    
    ollama.generate(MODEL_NAME, "You will receive frames. Summarize each in one sentence.")

   
    summaries = []
    for idx, image in enumerate(images, start=1):
        prompt = f"This is frame {idx} of the video. Summarize it in one sentence."
        try:
            response = ollama.generate(MODEL_NAME, prompt, images=[image])
            summaries.append(f"Frame {idx}: {response['response']}")
        except Exception as e:
            summaries.append(f"Frame {idx}: Error - {e}")

    
    summary_prompt = (
        "Here are frame summaries:\n" +
        "\n".join(summaries) +
        "\nNow summarize the entire video in a few sentences."
    )

    final_response = ollama.generate(MODEL_NAME, summary_prompt)
    final_summary = final_response['response']

    return ResponseBody(
        root=BatchTextResponse(texts=[
            TextResponse(value=final_summary, title="Final Video Summary")
        ])
    )
server.add_app_metadata(
    name= "Video Summarization",
    author="Priyanka",
    version= "1.0.0",
    info= "Video Summarization using Gemma model."
)


if __name__ == "__main__":
    server.run(debug=True)
