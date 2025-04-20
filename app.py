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
import whisper
import ollama

server = MLServer(__name__)

VIDEO_PATH = "video.mp4"
FRAME_FOLDER = "video_frames/"
AUDIO_PATH = "extracted_audio.wav"
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

def extract_audio_ffmpeg(video_path, audio_path=AUDIO_PATH):
    command = [
        "ffmpeg",
        "-i", video_path,
        "-q:a", "0",
        "-map", "a",
        audio_path,
        "-y"
    ]
    subprocess.run(command, check=True)

def transcribe_audio(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result['text']

@server.route(
    "/summarize",
    task_schema_func=create_video_summary_schema,
    short_title="Video Summarization",
    order=0
)
def summarize_video(inputs: Inputs, parameters: Parameters):  
    fps = parameters.get("fps", 1)

    # Step 1: Extract frames from the video
    extract_frames_ffmpeg(inputs["input_file"].path, FRAME_FOLDER, fps=fps)

    # Step 2: Extract audio and transcribe it
    extract_audio_ffmpeg(inputs["input_file"].path, AUDIO_PATH)
    transcribed_text = transcribe_audio(AUDIO_PATH)

    # Step 3: Prepare output paths
    out_path = Path(inputs["output_directory"].path)
    out_path_captions = str(out_path / f"captions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    out_path_summary = str(out_path / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    
    # Step 4: Describe each frame
    images = sorted([
        os.path.join(FRAME_FOLDER, f)
        for f in os.listdir(FRAME_FOLDER)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    summaries = []
    for idx, image in enumerate(images, start=1):
        prompt = f"This is frame {idx} of the video. Summarize it in one sentence."
        try:
            response = ollama.generate(MODEL_NAME, prompt, images=[image])
            summaries.append(f"Frame {idx}: {response['response']}")
        except Exception as e:
            summaries.append(f"Frame {idx}: Error - {e}")

    # Step 5: Summarize the whole video using both visual + audio data
    summary_prompt = (
        "Here are one-sentence descriptions of each frame of a video:\n" +
        "\n".join(summaries) +
        "\nHere is the transcribed audio from the video:\n" +
        transcribed_text +
        "\nSummarize the overall video in a few sentences using both visual and audio context."
    )

    final_response = ollama.generate(MODEL_NAME, summary_prompt)
    final_summary = final_response['response']

    # Step 6: Write results to files
    with open(out_path_captions, 'w', encoding='utf-8') as f:
        for line in summaries:
            f.write(line + '\n')

    with open(out_path_summary, 'w', encoding='utf-8') as f:
        f.write(final_summary.strip())

    # Step 7: Clean up temporary files
    shutil.rmtree(FRAME_FOLDER, ignore_errors=True)
    if os.path.exists(AUDIO_PATH):
        os.remove(AUDIO_PATH)

    return ResponseBody(FileResponse(path=out_path_summary, file_type="text"))

server.add_app_metadata(
    name="Video Summarization",
    author="Priyanka",
    version="1.0.0",
    info="Video Summarization using Gemma model with audio transcription."
)

if __name__ == "__main__":
    server.run(debug=True)
