from transformers import BitsAndBytesConfig, LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor
import torch
import av
import numpy as np
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from matplotlib import pyplot as plt
from matplotlib import animation
from IPython.display import HTML
import matplotlib as mpl
mpl.rcParams['animation.writer'] = 'ffmpeg'


# Configuration for 4-bit quantization
quantization_config = None


# Load processor and model
processor = LlavaNextVideoProcessor.from_pretrained("llava-hf/LLaVA-NeXT-Video-7B-hf")
model = LlavaNextVideoForConditionalGeneration.from_pretrained(
    "llava-hf/LLaVA-NeXT-Video-7B-hf",
    device_map='auto'  # CPU/GPU mapping
)

def read_video_pyav(container, indices, target_size=(224, 224)):
    """
    Decode the video with PyAV decoder.
    Args:
        container (av.container.input.InputContainer): PyAV container.
        indices (List[int]): List of frame indices to decode.
    Returns:
        np.ndarray: np array of decoded frames of shape (num_frames, height, width, 3).
    """
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    with tqdm(total=len(indices), desc="Reading frames") as pbar:
        for i, frame in enumerate(container.decode(video=0)):
            if i > end_index:
                break
            if i >= start_index and i in indices:
                frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

# Load and process video
video_path = "video.mp4"
container = av.open(video_path)
total_frames = container.streams.video[0].frames
indices = np.arange(0, total_frames, total_frames / 20).astype(int)
clip_car = read_video_pyav(container, indices)

# Visualize video frames (optional)
video = clip_car
fig = plt.figure()
im = plt.imshow(video[0, :, :, :])
plt.close()

def init():
    im.set_data(video[0, :, :, :])

def animate(i):
    im.set_data(video[i, :, :, :])
    return im

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=video.shape[0], interval=100)
HTML(anim.to_html5_video())

# Prepare conversation prompt
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What do you see in this video?"},
            {"type": "video"},
        ],
    },
]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

# Process inputs for the model
inputs = processor([prompt], videos=[clip_car], padding=True, return_tensors="pt").to(model.device)

# Generate response
generate_kwargs = {"max_new_tokens": 500, "do_sample": True, "top_p": 0.9}
output = model.generate(**inputs, **generate_kwargs)
generated_text = processor.batch_decode(output, skip_special_tokens=True)

# Print the generated text
print(generated_text)