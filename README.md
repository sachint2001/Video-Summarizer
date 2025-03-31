# Video-Summarizer (using BLIP+Bart)

Video-Summarizer is a tool used to generate summaries of videos. This model uses a combination of BLIP and BART models to summarize descriptions of videos. We first extract frames from videos and use BLIP to generate captions of the frames and then BART summarizes these captions to generate an overall summary of the video.

---

# Installation

## Clone the repository

```
git clone https://github.com/sachint2001/Video-Summarizer.git
cd Video-Summarizer
```

## Setup virtual environment

### Setup the virtual environment using:
```
python -m venv videosumm-env
```

### Activate it on Linux/MAC using:

```
source videosumm-env/bin/activate  
```

### Activate on Windows using: 

```
.\videosumm-env\Scripts\activate
```  
You might need to choose the appropriate file depending on the type of terminal you are running. Example: Powershell will require you to run activate.ps1

## Install dependencies

Install the required dependencies using:

```
pip install -r requirements.txt
```

## Setting the device to be used as GPU or CPU
TBC

# Usage

In the main function, modify the following variables as required:

- video_path = "video.mp4" (Path to the video you wish to summarize)
- keyframe_folder = "keyframes/" (Folder to save the frames extracted)
- captions_file = "frame_captions.txt" (File to save the captions generated)
- final_summary_file = "video_summary.txt" (File to save the video summary generated)

Run the code using:

```
python blipmodel.py
```
This should generate a summary in the file name used in the final_summary_file variable.

The first run will require internet access and some time to download the models from huggingface. After that everything should run locally.