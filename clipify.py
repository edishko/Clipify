import numpy as np
import face_recognition
from moviepy.editor import VideoFileClip, ImageSequenceClip, AudioFileClip, TextClip, CompositeVideoClip
import cv2
import time
from pytube import YouTube
from pytube.exceptions import PytubeError
import subprocess
import os
import ffmpeg
import yt_dlp as youtube_dl
from youtube_transcript_api import YouTubeTranscriptApi
import whisper
import textwrap
import dlib
from scipy.spatial import distance as dist
from openvino.runtime import Core

class Video:
    
    class Transcript:
    
        @staticmethod
        def get_whisper(input_video_path: str, whisper_model: str = 'medium'):
            if not os.path.exists(input_video_path):
                raise FileNotFoundError("Input video not found.")
        
            audio = whisper.load_audio(input_video_path)
            model = whisper.load_model(whisper_model)

            transcript = whisper.transcribe(model, audio, fp16=False)

            return transcript

        @staticmethod
        def get_youtube(video_id: str):
            try:
                transcript = YouTubeTranscriptApi.get_transcript(video_id)
                return transcript
            
            except Exception as e:
                print(f"Youtube Transcript Error: {e}")

        @staticmethod
        def text(transcript_json: dict):
            formatted_transcript = ''

            # Format transcript as text
            for entry in transcript:
                start_time = entry.get('start', 0)
                start_time_str = "{:.2f}".format(start_time)

                end_time = entry.get('end', start_time + entry.get('duration', 0))
                end_time_str = "{:.2f}".format(end_time)

                text = entry.get('text', '')

                formatted_transcript += f"{start_time_str} --> {end_time_str} : {text}\n"

            print('Transcript fetched successfully.')
            return formatted_transcript

    class Download:

        def youtube(url, output_path=""):
            ydl_opts = {
                'format': 'bestvideo[ext!=webm]+bestaudio[ext!=webm]/best[ext!=webm]',
                'outtmpl': '%(title)s.%(ext)s',
            }
            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

    class Edit:

        def __init__(self, video_path: str = None, video: VideoFileClip = None):
            
            if video_path is not None and video is not None:
                raise ValueError("Provide either 'video_path' or 'video', not both.")
            
            if video_path:
                self.video = VideoFileClip(video_path)
            elif video:
                self.video = video
            else:
                raise ValueError("Provide either 'video_path' or 'video'.")

        def crop(self, start_time: float, end_time: float = None):
            
            # Set the duration of the segment
            if end_time is not None:
                segment_clip = video.subclip(start_time, end_time)
            else:
                segment_clip = video.subclip(start_time)

            print("Segment created.")
            return segment_clip
            
        def clip(input_video_path: str, output_video_path: str, face_check_interval: int = 1, smooth_frames_alpha: float = 0.2, smooth_coordinates_alpha: float = 0.35, clip_aspect_ratio: float = 9 / 16, confidence_threshold: float = 0.76):
    
            def smooth_frames(current_frame: np.ndarray, previous_frame: np.ndarray = None, alpha: float = 0.1):
                """
                Smooths the current frame by blending it with the previous frame.

                Args:
                    current_frame (numpy.ndarray): The current frame to be smoothed.
                    previous_frame (numpy.ndarray, optional): The previous frame to blend with the current frame.
                    alpha (float, optional): The blending factor. Higher alpha values give more weight to the current frame.

                Returns:
                    numpy.ndarray: The smoothed frame.
                """
                if previous_frame is None:
                    # If there is no previous frame, return the current frame as is
                    return current_frame
                else:
                    # Ensure both frames have the same dimensions
                    previous_frame_resized = cv2.resize(previous_frame, (current_frame.shape[1], current_frame.shape[0]))

                    # Perform the blending operation
                    smoothed_frame = cv2.addWeighted(previous_frame_resized, alpha, current_frame, 1 - alpha, 0)
                    return smoothed_frame
            
            def smooth_coordinates(current_coordinates: (int, int), previous_coordinates: (int, int) = None, alpha: float = 0.4):
                """
                Smooths the current coordinates based on the previous coordinates using a specified smoothing factor.

                Args:
                    current_coordinates (tuple of int): The current coordinates (x, y) of the point to be smoothed.
                    previous_coordinates (tuple of int, optional): The previous coordinates (x, y) of the point.
                    alpha (float, optional): The smoothing factor. Higher alpha values give more weight to the current coordinates.

                Returns:
                    tuple of int: The smoothed coordinates (x, y).
                """
                if previous_coordinates is None:
                    # If there are no previous coordinates, return the current coordinates as is
                    return current_coordinates
                else:
                    # Calculate the smoothed coordinates based on the weighted sum of current and previous coordinates
                    smoothed_x = int(alpha * current_coordinates[0] + (1 - alpha) * previous_coordinates[0])
                    smoothed_y = int(alpha * current_coordinates[1] + (1 - alpha) * previous_coordinates[1])
                    return smoothed_x, smoothed_y

            def model_init(model: str, device: str = "CPU"):
                """
                Initialize the OpenVINO model for face detection.

                Args:
                    model (str): Path to the XML file of the model.
                    device (str, optional): Device to run inference on (e.g., "CPU", "GPU"). Defaults to "CPU".

                Returns:
                    Tuple: Tuple containing the initialized model, input layer, and output layer.
                """
                try:
                    # Load OpenVINO Core
                    core = Core()

                    # Read model from XML file
                    detection_model = core.read_model(model = model)

                    # Compile model for the specified device
                    compiled_model = core.compile_model(model = detection_model, device_name = device)

                    # Get input and output layers
                    input_layer = compiled_model.input(0)
                    output_layer = compiled_model.output(0)

                    return compiled_model, input_layer, output_layer

                except Exception as e:
                    print(f"Error initializing model: {e}")
                    return None, None, None

            def face_detection(frame: np.ndarray, input_layer, output_layer, confidence_threshold: float = 0.76):
                """
                Perform face detection on a single frame.

                Args:
                    frame (np.ndarray): Input frame.
                    input_layer: Input layer of the model.
                    output_layer: Output layer of the model.
                    confidence_threshold (float, optional): Minimum confidence threshold for detected faces. Defaults to 0.76.

                Returns:
                    List: List of detected face center coordinates [(x1, y1), (x2, y2), ...].
                """
                try:
                    # Resize frame to match model input size
                    resized_frame = cv2.resize(src=frame, dsize=(input_layer.shape[3], input_layer.shape[2]))
                    input_data = np.expand_dims(np.transpose(resized_frame, (2, 0, 1)), 0).astype(np.float32)

                    # Perform inference
                    request = compiled_model.create_infer_request()
                    request.infer(inputs={input_layer.any_name: input_data})
                    result = request.get_output_tensor(output_layer.index).data

                    frame_height, frame_width = frame.shape[:2]
                    detected_faces = []

                    # Iterate over detections
                    for detection in result[0][0]:
                        label = int(detection[1])
                        confidence = float(detection[2])

                        # Check if confidence meets threshold
                        if confidence > confidence_threshold:
                            xmin, ymin = int(detection[3] * frame_width), int(detection[4] * frame_height)
                            xmax, ymax = int(detection[5] * frame_width), int(detection[6] * frame_height)

                            # Calculate the center of the detected face
                            face_center = ((xmax - xmin) // 2 + xmin, (ymax - ymin) // 2 + ymin)
                            detected_faces.append(face_center)

                    return detected_faces

                except Exception as e:
                    print(f"Error in face detection: {e}")
                    return []

            def frame_portrait(frame: np.ndarray, detected_faces: list, clip_shape: (int, int), previous_frame: np.ndarray = None, previous_face_center: (int, int) = None, smooth_frames_alpha: float = 0.1, smooth_coordinates_alpha: float = 0.4):
                """
                Process frames in portrait orientation.

                Args:
                    frame (np.ndarray): Input frame.
                    detected_faces (list): List of detected face center coordinates [(x1, y1), (x2, y2), ...].
                    clip_shape (tuple): Tuple containing the width and height of the clip.
                    previous_frame (np.ndarray, optional): Previous frame for smoothing. Defaults to None.
                    previous_face_center (tuple of int, optional): Previous coordinates of the detected face. Defaults to None.
                    smooth_frames_alpha (float, optional): The blending factor for smoothing frames. Defaults to 0.1.
                    smooth_coordinates_alpha (float, optional): The smoothing factor for face coordinates. Defaults to 0.4.

                Returns:
                    tuple: Processed frame, orientation, and smoothed face center.
                """
                orientation = "portrait"

                clip_width, clip_height = clip_shape
                
                if detected_faces:
                    face_center = detected_faces[0]
                    smoothed_face_center = smooth_coordinates(face_center, previous_face_center, smooth_coordinates_alpha)
                
                    crop_x = max(0, smoothed_face_center[0] - clip_width // 2)
                    crop_y = max(0, smoothed_face_center[1] - clip_height // 2)

                elif previous_face_center is not None:
                    smoothed_face_center = previous_face_center

                    crop_x = max(0, smoothed_face_center[0] - clip_width // 2)
                    crop_y = max(0, smoothed_face_center[1] - clip_height // 2)

                else:
                    smoothed_face_center = None

                    crop_x = clip_width // 2
                    crop_y = clip_height // 2

                # Crop the frame around the detected face if faces are detected
                
                cropped_frame = frame[crop_y:crop_y + clip_height, crop_x:crop_x + clip_width]
                

                smoothed_frame = smooth_frames(cropped_frame, previous_frame, smooth_frames_alpha)

                # Resize cropped frame without changing aspect ratio
                finished_frame = cv2.resize(smoothed_frame, (clip_width, clip_height), interpolation=cv2.INTER_LINEAR)
                
                return finished_frame, orientation, smoothed_face_center 

            def frame_landscape(frame: np.ndarray, clip_shape: (int, int), previous_frame: np.ndarray = None, previous_face_center: (int, int) = None, smooth_frames_alpha: float = 0.1, smooth_coordinates_alpha: float = 0.4):
                """
                Process frames in landscape orientation.

                Args:
                    frame (np.ndarray): Input frame.
                    clip_shape (tuple): Tuple containing the width and height of the clip.
                    previous_frame (np.ndarray, optional): Previous frame for smoothing. Defaults to None.
                    previous_face_center (tuple of int, optional): Previous coordinates of the detected face. Defaults to None.
                    smooth_frames_alpha (float, optional): The blending factor for smoothing frames. Defaults to 0.1.
                    smooth_coordinates_alpha (float, optional): The smoothing factor for face coordinates. Defaults to 0.4.

                Returns:
                    tuple: Processed frame, orientation, and None for smoothed face center.
                """
                orientation = "landscape"

                clip_width, clip_height = clip_shape

                # Process frames with multiple faces
                landscape_clip_width = clip_width
                landscape_clip_height = int(clip_width * clip_width / clip_height)
                
                landscape_frame = cv2.resize(frame, (landscape_clip_width, landscape_clip_height), interpolation=cv2.INTER_LINEAR)
                blank_frame = np.zeros((clip_height, clip_width, 3), dtype=np.uint8)

                x_offset = (clip_width - landscape_clip_width) // 2
                y_offset = (clip_height - landscape_clip_height) // 2

                blank_frame[y_offset:y_offset + landscape_clip_height, x_offset:x_offset + landscape_clip_width] = landscape_frame
                smoothed_frame = smooth_frames(blank_frame, previous_frame, smooth_frames_alpha)
                finished_frame = smoothed_frame
                
                return finished_frame, orientation, None

            # Load the input video
            input_video = VideoFileClip(input_video_path)

            # Get video properties
            video_width, video_height = input_video.size
            fps = input_video.fps
            duration = input_video.duration

            # Determine clip dimensions
            clip_width, clip_height = int(video_height * clip_aspect_ratio), video_height

            # Initialize Model
            compiled_model, input_layer, output_layer = model_init(model = "intel/face-detection-0202/FP32/face-detection-0202.xml", device = "CPU")
            print("Model initialized successfully:", compiled_model is not None)
            
            # Initialize variables
            video_frames = []
            
            frame_orientations = []

            previous_frame = None
            previous_face_center = None

            # Process each frame in the input video
            for index, frame in enumerate(input_video.iter_frames(fps=fps, dtype="uint8")):
                
                if index % face_check_interval == 0:
        
                    detected_faces = face_detection(frame = frame, input_layer = input_layer, output_layer = output_layer, confidence_threshold = confidence_threshold)

                if len(detected_faces) >= 2:
                    finished_frame, orientation, smoothed_coordinates = frame_landscape(frame=frame, clip_shape=(clip_width, clip_height), previous_frame=previous_frame, smooth_frames_alpha=smooth_frames_alpha, smooth_coordinates_alpha=smooth_coordinates_alpha)   
                
                else:
                    finished_frame, orientation, smoothed_coordinates = frame_portrait(frame=frame, detected_faces=detected_faces, clip_shape=(clip_width, clip_height), previous_frame=previous_frame, previous_face_center=previous_face_center, smooth_frames_alpha=smooth_frames_alpha, smooth_coordinates_alpha=smooth_coordinates_alpha)
                
                previous_frame = finished_frame
                previous_face_center = smoothed_coordinates
                

                video_frames.append(finished_frame)
                # frame_orientations.append((index, frame, orientation))
            
                # print((index, frame.shape, orientation))

            # Write the processed frames to a new video file
            output_video = ImageSequenceClip(video_frames, fps=fps)
            output_video.audio = input_video.audio.subclip(0, duration)
            output_video = output_video.subclip(0, duration)
            output_video.write_videofile(output_video_path, codec="libx264", audio_codec="aac", temp_audiofile="temp-audio.m4a", remove_temp=True, fps=fps)

        def caption(input_video_path: str, output_video_path: str, transcript: dict = None, max_chars_per_line: int = 30, max_words_per_phrase: int = 4, fontsize: int = 70, font: str = 'Arial-Bold', stroke_color: str ='black', stroke_width: int = 1, position: tuple = ('center', 0.6)):
            
            def wrap_text(text: str, max_chars_per_line: int, max_width: int):
                lines = textwrap.wrap(text, width=max_chars_per_line)
                wrapped_lines = []

                for line in lines:
                    if len(line) > max_width:
                        # If the line is longer than the maximum width, wrap it further
                        wrapped_lines.extend(textwrap.wrap(line, width=max_width))
                    else:
                        wrapped_lines.append(line)

                return '\n'.join(wrapped_lines)

            def split_text_into_phrases(text: str, max_words_per_phrase: int):
                words = text.split()
                phrases = [words[i:i + max_words_per_phrase] for i in range(0, len(words), max_words_per_phrase)]
                return [' '.join(phrase) for phrase in phrases]

            if transcript is None:
                transcript = Video.Transcript.get_whisper(input_video_path)

            input_video = VideoFileClip(input_video_path)  # Load the video clip

            subtitle_clips = []

            # Get the maximum width of the video
            max_width = input_video.size[0]

            for segment in transcript['segments']:
                start_time = segment["start"]
                end_time = segment["end"]
                text = segment["text"]

                wrapped_text = wrap_text(text, max_chars_per_line, max_width)
                phrases = split_text_into_phrases(wrapped_text, max_words_per_phrase)

                total_duration = end_time - start_time

                for phrase in phrases:
                    phrase_duration = max((total_duration / len(phrases)), 1)
                    text_clip = TextClip(phrase, fontsize=fontsize, color='white', font=font, stroke_color=stroke_color, stroke_width=stroke_width)
                    text_clip = text_clip.set_position(position, relative=True).set_start(start_time).set_end(start_time + min(phrase_duration, total_duration))
                    subtitle_clips.append(text_clip)
                    start_time += min(phrase_duration, total_duration)
                    total_duration -= min(phrase_duration, total_duration)

            output_video = CompositeVideoClip([input_video] + subtitle_clips)
            output_video.write_videofile(output_video_path, codec="libx264", audio_codec="aac")

    class AI:
        pass

start = time.perf_counter()
# Video.Download.youtube(url="https://www.youtube.com/watch?v=1aA1WGON49E")
# transcript = Video.Transcript.get_youtube("RnjTYBhAcfA")
Video.Edit.clip("tren2.mp4", "negga3.mp4")
# print(Video.Transcript.text(transcript))
end = time.perf_counter()
print(f"{end-start} Seconds!")