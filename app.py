import argparse
import logging
from io import BytesIO
import wave
import os

import requests
import numpy as np
import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader
from moviepy.editor import (
    VideoClip,
    AudioClip, 
    AudioFileClip,
    ImageClip, 
    TextClip, 
    CompositeVideoClip, 
    concatenate_audioclips, 
    concatenate_videoclips
)
from moviepy.audio.AudioClip import AudioArrayClip
from moviepy.video.fx import (
    resize, mirror_x
)

logging.basicConfig(level=logging.INFO)
SPEAKER_ID = 47
SYNTHESIS_SPEAKER_URL = "http://localhost:50021"
RATE = 1.25

def parse():
    parser = argparse.ArgumentParser(
        description="Instructor Script to Movie Script"
    )

    parser.add_argument(
        "-i", 
        "--input", 
        type=str, 
        dest="input",
        help="Input an instructor script yaml path.",
        default="script.yaml"
    )

    args = parser.parse_args()

    print("yaml path input:", args.input)
    return args

def generate_synthesis(text: str):
    output_path = f"artifacts/{text}.wav"

    # 既にあるのであれば読み込む
    if os.path.exists(output_path):
        with open(output_path, "rb") as fr:
            synthesis_voice = fr.read()
        return synthesis_voice

    # voicevox に音声合成クエリを生成してもらう
    headers = {"Content-Type": "application/json"}
    url_params = {"speaker": SPEAKER_ID, "text": text}
    res = requests.post(
        f"{SYNTHESIS_SPEAKER_URL}/audio_query", 
        headers=headers,
        params=url_params
    )
    audio_query = res.json()

    # 
    url_params = {"speaker": SPEAKER_ID}
    res = requests.post(
        f"{SYNTHESIS_SPEAKER_URL}/synthesis", 
        headers=headers,
        params=url_params,
        json=audio_query
    )

    synthesis_voice = res.content
    with open(output_path, "wb") as fw:
        fw.write(synthesis_voice)
    return synthesis_voice


def audio_to_clip(audio_data) -> AudioClip:
    # 与えられたAudioデータをwaveに変換する
    audio_file = wave.open(BytesIO(audio_data), "rb")
    n_channels = audio_file.getnchannels()
    n_frames = audio_file.getnframes()
    framerate = audio_file.getframerate() * 2
    logging.info(f"n_channels:{n_channels}, n_frames:{n_frames}, framerate:{framerate}")

    # 読み込んだwave形式をnumpyに変換
    audio_numpy = np.frombuffer(audio_file.readframes(n_frames), dtype=np.int16)
    audio_numpy = audio_numpy.astype(np.float32) / (2 ** 15)  # Normalize int16 to float32 between -1 and 1
    audio_numpy = audio_numpy.reshape(-1, 1)

    if n_channels == 1:
        audio_numpy = audio_numpy.reshape(-1, 1)
    else:
        audio_numpy = audio_numpy.reshape(-1, n_channels)

    # AudioClipに変換
    audio_clip = AudioArrayClip(audio_numpy, fps=framerate)
    audio_clip.duration = audio_clip.duration * 2.0
    logging.info(f"audio duration: {audio_clip.duration}")

    return audio_clip


def audio_file_to_clip(filepath: str) -> AudioClip:
    # with AudioFileClip(filepath) as audio:
    #     audio_clip = audio.copy()
    # audio_clip.duration = audio_clip.duration + 0.25
    return AudioFileClip(filepath)


def build_images_for_movie():
    # 画像ファイルを読み込む
    background = ImageClip("images/background.png")
    character = ImageClip("images/character.png")
    frame = ImageClip("images/frame.png")

    # キャラクター画像とフレーム画像の表示位置を設定する
    character = character.set_position(("left", "top")).fx(mirror_x.mirror_x, ["mask"])
    frame = frame.set_position(("center", "bottom")).fx(resize.resize, 1.3)
    
    return background, character, frame


def generate_text_from_audio(script: str, audio_clip: AudioClip, start_at: float = 0.0) -> TextClip:
    # 台詞を生成する
    text_clip = TextClip(
        script, fontsize=44, color="black", font="Noto-Sans-CJK-JP", 
        method="caption", size=(1200, 250), align="West")
    duration = audio_clip.duration
    text_clip = (
        text_clip
        .set_position((0.2,0.8), relative=True) # .set_position(("center", "bottom"))
        .set_duration(duration)
        # .set_start(start_at)
        # .set_end(duration)
    )
    logging.info(f"text duration: {text_clip.duration}")
    return text_clip


def merge_texts_and_audios(
    text_clips: list[TextClip], 
    audio_clips: list[AudioClip], 
    title: str, 
    explain_text_clip: list[TextClip] | None = None
):
    # 画像ファイルを読み込む
    background, character, frame = build_images_for_movie()

    # 入力の結合
    total_audio_clip = concatenate_audioclips(audio_clips)
    logging.info(f"total_audio_clip: {total_audio_clip.duration}")

    # 各テキストクリップに対応する音声クリップのdurationを設定
    synced_clips = []
    for i, (text_clip, audio_clip) in enumerate(zip(text_clips, audio_clips)):
        composed_clips = [background, character, frame, text_clip]
        if explain_text_clip is not None and explain_text_clip[i] is not None:
            composed_clips.append(explain_text_clip[i])

        synced_clip = (
            CompositeVideoClip(composed_clips)
            .set_duration(audio_clip.duration)
            # .set_end(audio_clip.end)
            .set_audio(audio_clip)
        )
        logging.info((synced_clip.duration, synced_clip.start, synced_clip.end))
        synced_clips.append(synced_clip)

    video = concatenate_videoclips(synced_clips)
    if video.duration is None:
        video.duration = total_audio_clip.duration
    
    # video = video.set_audio(total_audio_clip)

    # タイトルの文字を追加
    title_clip = (
        TextClip(
            title, fontsize=64, color="white", font="Noto-Sans-CJK-JP", 
            method="caption", size=(1000, 200), align="West"
        )
        .set_position((0.4,0.0025), relative=True)
        .set_duration(video.duration)
    )

    video = CompositeVideoClip([video, title_clip])

    return video


def generate_exercise_explain_clip(answers: list[str], audio_clip: AudioClip):
    explain_text = "\n".join([f"{idx+1}. {ans}" for idx, ans in enumerate(answers)])
    title_clip = (
        TextClip(
            explain_text, fontsize=32, color="white", font="Noto-Sans-CJK-JP", 
            method="caption", size=(1000, 640), align="West"
        )
        .set_position((0.4,0.05), relative=True)
        .set_duration(audio_clip.duration)
    )

    return title_clip


def generate_voice_and_clip(script):
    synthesis_voice = generate_synthesis(script)
    audio_clip = audio_file_to_clip(f"artifacts/{script}.wav")
    # audio_clip = audio_to_clip(synthesis_voice)
    text_clip = generate_text_from_audio(script, audio_clip)
    return audio_clip, text_clip


def generate_lecture_clip(scripts: list[str], title: str):
    audio_clips = []
    text_clips = []
    for script in scripts:
        audio_clip, text_clip = generate_voice_and_clip(script)
        audio_clips.append(audio_clip)
        text_clips.append(text_clip)
        # script_pos += text_clip.duration

    video = merge_texts_and_audios(text_clips, audio_clips, title)
    return video


def generate_exercise_clip(scripts: list[str], title: str, exercise: dict):
    audio_clips = []
    text_clips = []
    # script生成部
    for script in scripts:
        audio_clip, text_clip = generate_voice_and_clip(script)
        audio_clips.append(audio_clip)
        text_clips.append(text_clip)
        # script_pos += text_clip.duration

    # exercise
    explain_text_clips = [None] * len(scripts)
    answers = exercise["answers"]

    question = exercise["question"]
    audio_clip, text_clip = generate_voice_and_clip(question)
    audio_clips.append(audio_clip)
    text_clips.append(text_clip)
    explain_clip = generate_exercise_explain_clip(answers, audio_clip)
    explain_text_clips.append(explain_clip)

    correct_idx = exercise["correct"]
    answer = f"正解は{correct_idx+1}番です。"
    audio_clip, text_clip = generate_voice_and_clip(answer)
    audio_clips.append(audio_clip)
    text_clips.append(text_clip)
    explain_clip = generate_exercise_explain_clip(answers, audio_clip)
    explain_text_clips.append(explain_clip)

    video = merge_texts_and_audios(text_clips, audio_clips, title, explain_text_clips)
    return video

def merge_and_write_video(videos: list[VideoClip]):
    video = concatenate_videoclips(videos)
    video.write_videofile("output.mp4", fps=24)


def main():
    args = parse()
    filepath = args.input
    with open(filepath, "r") as fr:
        instructor_lectures = yaml.load(fr, Loader)
    
    logging.info(instructor_lectures)

    videos = []

    for lecture in instructor_lectures:
        title = lecture["topic"]
        lecture_case = lecture["case"]
        scripts = lecture["scripts"]
        if lecture_case == "lecture":
            video = generate_lecture_clip(scripts, title)
        if lecture_case == "exercise":
            exercise = lecture["exercise"]
            video = generate_exercise_clip(scripts, title, exercise)
        
        videos.append(video)
        
    
    merge_and_write_video(videos)


if __name__ == "__main__":
    main()