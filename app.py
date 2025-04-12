import os
import subprocess
import shutil
from typing import TypedDict
from flask_ml.flask_ml_server import MLServer
from flask_ml.flask_ml_server.models import (
    ParameterSchema,
    TaskSchema,
    ResponseBody,
    TextResponse,
    BatchTextResponse,
    FileInput,
    InputSchema,
    InputType,
    DirectoryInput,
    FileResponse
)
from datetime import datetime
from pathlib import Path

import ollama

server = MLServer(__name__)

VIDEO_PATH = "video.mp4"
FRAME_FOLDER = "video_frames/"
MODEL_NAME = "gemma3:4b"

class Inputs(TypedDict):
    input_file: FileInput
    output_directory: DirectoryInput

class Parameters(TypedDict):
    fps: int  

from flask_ml.flask_ml_server.models import IntParameterDescriptor

def create_video_summary_schema() -> TaskSchema:
    input_schema = InputSchema(
        key="input_file",
        label="Video file to summarize",
        input_type=InputType.FILE,
    )
    output_schema = InputSchema(
        key="output_directory",
        label="Path to save results",
        input_type=InputType.DIRECTORY,
    )
    fps_param_schema = ParameterSchema(
        key="fps",
        label="Frame Rate (fps)",
        subtitle="Set how many frames per second to extract from the video",
        value=IntParameterDescriptor(default=1, min=1, max=30),
    )

    return TaskSchema(inputs=[input_schema, output_schema], parameters=[fps_param_schema])

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

    extract_frames_ffmpeg(inputs["input_file"].path, FRAME_FOLDER, fps=fps)
    out_path = Path(inputs["output_directory"].path)
    out_path_captions = str(out_path / f"captions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    out_path_summary = str(out_path / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    
    images = sorted([
        os.path.join(FRAME_FOLDER, f)
        for f in os.listdir(FRAME_FOLDER)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    ollama.generate(MODEL_NAME, "You will receive frames from a video in sequence, one at a time. For each frame, generate a concise one-sentence description.")

    summaries = []
    for idx, image in enumerate(images, start=1):
        prompt = f"This is frame {idx} of the video. Summarize it in one sentence."
        try:
            response = ollama.generate(MODEL_NAME, prompt, images=[image])
            summaries.append(f"Frame {idx}: {response['response']}")
        except Exception as e:
            summaries.append(f"Frame {idx}: Error - {e}")

    summary_prompt = (
        "Here are one-sentence descriptions of each frame of a video:\n" +
        "\n".join(summaries) +
        "\nSummarize the overall video in a few sentences. Keep in mind that certain frames occuring one after the other could be describing the same incident that has just occured."
    )

    final_response = ollama.generate(MODEL_NAME, summary_prompt)
    final_summary = final_response['response']

    with open(out_path_captions, 'w', encoding='utf-8') as f:
        for line in summaries:
            f.write(line + '\n')

    with open(out_path_summary, 'w', encoding='utf-8') as f:
        f.write(final_summary.strip())

    shutil.rmtree(FRAME_FOLDER, ignore_errors=True)

    return ResponseBody(FileResponse(path=out_path_summary, file_type="text"))

server.add_app_metadata(
    name="Video Summarization",
    author="Priyanka",
    version="1.0.0",
    info="Video Summarization using Gemma model."
)

if __name__ == "__main__":
    server.run(debug=True)
