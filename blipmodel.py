import cv2
import os
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
from PIL import Image

def extract_keyframes(video_path, keyframe_folder, interval_seconds=2):
    """Extracts keyframes from a video at specified intervals."""
    cap = cv2.VideoCapture(video_path)
    os.makedirs(keyframe_folder, exist_ok=True)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(frame_rate * interval_seconds)
    frame_count = 0
    keyframe_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            keyframe_path = os.path.join(keyframe_folder, f"keyframe_{keyframe_count}.jpg")
            cv2.imwrite(keyframe_path, frame)
            keyframe_count += 1
        frame_count += 1

    cap.release()
    print(f"Extracted {keyframe_count} keyframes and saved in {keyframe_folder}")

def generate_image_captions(keyframe_folder):
    """Generates captions for images in a folder using BLIP."""
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    keyframes = sorted(os.listdir(keyframe_folder))
    captions = {}

    for keyframe in keyframes:
        image_path = os.path.join(keyframe_folder, keyframe)
        image = Image.open(image_path).convert("RGB")
        inputs = processor(image, return_tensors="pt")
        caption_ids = model.generate(**inputs)
        caption = processor.batch_decode(caption_ids, skip_special_tokens=True)[0]
        captions[keyframe] = caption
        print(f"{keyframe}: {caption}")
    return captions

def summarize_captions(captions_file):
    """Summarizes captions from a text file using BART."""
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summarizer.model.config.max_position_embeddings = 8192

    with open(captions_file, "r") as f:
        captions_text = f.read()

    def chunk_text(text, max_tokens=800):
        tokenizer = summarizer.tokenizer
        tokens = tokenizer.tokenize(text)
        num_chunks = len(tokens) // max_tokens + 1
        chunks = [tokenizer.convert_tokens_to_string(tokens[i * max_tokens:(i + 1) * max_tokens]) for i in range(num_chunks)]
        return chunks

    chunks = chunk_text(captions_text, max_tokens=800)
    summaries = [summarizer(chunk, max_length=90, min_length=30, do_sample=False, truncation=True)[0]['summary_text'] for chunk in chunks]
    final_summary = summarizer(" ".join(summaries), max_length=100, min_length=30, do_sample=False, truncation=True)[0]['summary_text']
    print("\nFinal Video Summary:")
    print(final_summary)
    return final_summary

def main():
    """Main function to orchestrate video processing and summarization."""
    video_path = "t1.mp4"
    keyframe_folder = "keyframes2"
    captions_file = "video_summary2.txt"
    final_summary_file = "final_video_summary2.txt"

    extract_keyframes(video_path, keyframe_folder)
    captions = generate_image_captions(keyframe_folder)

    with open(captions_file, "w") as f:
        for keyframe, caption in captions.items():
            f.write(f"{keyframe}: {caption}\n")
    print(f"\nSummary saved to {captions_file}")

    final_summary = summarize_captions(captions_file)

    with open(final_summary_file, "w") as f:
        f.write(final_summary)
    print(f"\nFinal summary saved to {final_summary_file}")

if __name__ == "__main__":
    main()

# Install necessary libraries if not already installed
# !pip install transformers torch torchvision opencv-python Pillow