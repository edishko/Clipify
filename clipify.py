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
import itertools
import ollama
import openai
import json

# take configs from a file
with open('config.json', 'r') as config:
    CONFIG = json.load(config)

CONFIG_AI = CONFIG.get("ai")

class Video:
    """
    Utility class containing various video functions.
    """
    class Transcript:
        """
        Utility class for handling video transcripts.
        """

        @staticmethod
        def get_whisper(input_video_path: str, whisper_model: str = 'medium') -> str:
            """
            Get transcript from a local video file using Whisper library.

            Args:
                input_video_path (str): Path to the input video file.
                whisper_model (str, optional): Whisper model to use for transcription. Defaults to 'medium'.

            Returns:
                str: Transcript of the video.
            
            Raises:
                FileNotFoundError: If the input video file is not found.
            """
            if not os.path.exists(input_video_path):
                raise FileNotFoundError("Input video not found.")
            
            audio = whisper.load_audio(input_video_path)
            model = whisper.load_model(whisper_model)

            transcript = whisper.transcribe(model, audio, fp16=False)

            return transcript

        @staticmethod
        def get_youtube(video_id: str) -> list:
            """
            Get transcript from a YouTube video using the YouTubeTranscriptApi.

            Args:
                video_id (str): YouTube video ID.

            Returns:
                list: List of transcript entries.

            Raises:
                Exception: If an error occurs while fetching the transcript.
            """
            try:
                transcript = YouTubeTranscriptApi.get_transcript(video_id)
                return transcript
                
            except Exception as e:
                print(f"Youtube Transcript Error: {e}")

        @staticmethod
        def get_text(transcript_json: dict) -> str:
            """
            Format transcript JSON into text.

            Args:
                transcript_json (dict): Transcript data in JSON format.

            Returns:
                str: Formatted transcript text.
            """
            formatted_transcript = ''

            # Format transcript as text
            for entry in transcript_json:
                start_time = entry.get('start', 0)
                start_time_str = "{:.2f}".format(start_time)

                end_time = entry.get('end', start_time + entry.get('duration', 0))
                end_time_str = "{:.2f}".format(end_time)

                text = entry.get('text', '')

                formatted_transcript += f"{start_time_str} --> {end_time_str} : {text}\n"

            print('Transcript fetched successfully.')
            return formatted_transcript

    class Download:
        """
        Utility class for downloading videos from various sources.
        """
        @staticmethod
        def youtube(url: str, output_path: str = ""):
            """
            Download a video from YouTube.

            Args:
                url (str): The URL of the YouTube video to download.
                output_path (str, optional): The output path where the downloaded video will be saved.
                    If not provided, the video will be saved in the current directory using the video title as the filename.

            Returns:
                None

            Raises:
                youtube_dl.utils.DownloadError: If there is an error during the download process.
            """
            ydl_opts = {
                'format': 'bestvideo[ext!=webm]+bestaudio[ext!=webm]/best[ext!=webm]',
                'outtmpl': output_path or '%(title)s.%(ext)s',
            }
            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

    class Edit:
        """
        Class for video editing operations.
        """
        @staticmethod
        def crop(input_video_path: str, output_video_path: str, start_time: float, end_time: float = None):
            """
            Crop a segment from a video file.

            Args:
                input_video_path (str): Path to the input video file.
                output_video_path (str): Path to save the cropped video file.
                start_time (float): Start time of the segment to crop, in seconds.
                end_time (float, optional): End time of the segment to crop, in seconds.
                    If not provided, the segment will extend to the end of the original video.

            Returns:
                None
            """
            # Load the input video
            input_video = VideoFileClip(input_video_path)

            # Set the duration of the segment
            if end_time is not None:
                segment_clip = input_video.subclip(start_time, end_time)
            else:
                segment_clip = input_video.subclip(start_time)

            # Write the cropped segment to the output file
            segment_clip.write_videofile(output_video_path)

            print("Segment created.")
        
        @staticmethod
        def clip(input_video_path: str, output_video_path: str, face_check_interval: int = 1, smooth_frames_alpha: float = 0.2, smooth_coordinates_alpha: float = 0.35, clip_aspect_ratio: float = 9 / 16, confidence_threshold: float = 0.76):
            """
            Clip a video with face orientation detection and processing.

            Args:
                input_video_path (str): Path to the input video file.
                output_video_path (str): Path to save the processed video file.
                face_check_interval (int, optional): Interval for checking faces in frames. Defaults to 1.
                smooth_frames_alpha (float, optional): The blending factor for smoothing frames. Defaults to 0.2.
                smooth_coordinates_alpha (float, optional): The smoothing factor for face coordinates. Defaults to 0.35.
                clip_aspect_ratio (float, optional): Aspect ratio of the output clip. Defaults to 9 / 16.
                confidence_threshold (float, optional): Minimum confidence threshold for detected faces. Defaults to 0.76.

            Returns:
                None
            """
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
                    tuple: Processed frame, smoothed face center.
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
                
                return finished_frame, smoothed_face_center 

            def frame_landscape(frame: np.ndarray, clip_shape: (int, int), previous_frame: np.ndarray = None, previous_face_center: (int, int) = None, smooth_frames_alpha: float = 0, smooth_coordinates_alpha: float = 0.4):
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
                    tuple: Processed frame, smoothed face center.
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
                
                return finished_frame, None

            def find_consecutive_frames(frame_orientations: list):
                """
                Find consecutive frames with the same orientation.

                Args:
                    frame_orientations (list): List of tuples containing frame index and orientation.

                Returns:
                    list: List of tuples, each containing the orientation, count of consecutive frames, and indices of consecutive frames.
                """
                if not frame_orientations:
                    return []

                consecutive_frames = []
                current_count = 1
                current_orientation = frame_orientations[0][1]
                start_index = 0

                for i in range(1, len(frame_orientations)):
                    if frame_orientations[i][1] != current_orientation:
                        consecutive_frames.append((current_orientation, current_count, list(range(start_index, start_index + current_count))))
                        current_orientation = frame_orientations[i][1]
                        current_count = 1
                        start_index = i
                    else:
                        current_count += 1

                consecutive_frames.append((current_orientation, current_count, list(range(len(frame_orientations) - current_count, len(frame_orientations)))))

                return consecutive_frames
            
            # Initialize Model
            compiled_model, input_layer, output_layer = model_init(model="intel/face-detection-0202/FP32/face-detection-0202.xml", device="GPU")
            print("Model initialized successfully:", compiled_model is not None)

            # Process the video
            with VideoFileClip(input_video_path) as input_video:
                fps = input_video.fps
                duration = input_video.duration
                
                input_width, input_height = input_video.size
                # Determine clip dimensions
                clip_width, clip_height = int(input_height * clip_aspect_ratio), input_height

                print(clip_width, clip_height)
                # Initialize variables
                frame_orientations = []

                # Process each frame in the input video
                current_frame_index = 0
                for frame in input_video.iter_frames(fps=fps, dtype="uint8"):
                    if current_frame_index % face_check_interval == 0:
                        detected_faces = face_detection(frame=frame, input_layer=input_layer, output_layer=output_layer, confidence_threshold=confidence_threshold)

                    if len(detected_faces) >= 2:
                        orientation = "landscape"
                    else:
                        orientation = "portrait"

                    frame_orientations.append((current_frame_index, orientation))
                    current_frame_index += 1

                # Find consecutive frames
                consecutive_frames = find_consecutive_frames(frame_orientations)

                # Write the processed frames to a new video file
                with VideoFileClip(input_video_path) as input_video:
                    new_video = []
                    for orientation, count, indices in consecutive_frames:
                        duration = count / fps

                        # Determine the correct orientation
                        if duration < 0.5:
                            orientation = "landscape" if orientation == "portrait" else "portrait"

                        # Process frames in batches
                        previous_frame = None
                        previous_face_center = None
                        current_frame_index = 0
                        for frame in input_video.iter_frames(fps=fps, dtype="uint8"):
                            if current_frame_index in indices:
                                if orientation == "portrait":
                                    detected_faces = face_detection(frame=frame, input_layer=input_layer, output_layer=output_layer, confidence_threshold=confidence_threshold)
                                    finished_frame, smoothed_coordinates = frame_portrait(frame=frame, detected_faces=detected_faces, clip_shape=(clip_width, clip_height), previous_frame = previous_frame, previous_face_center = previous_face_center, smooth_frames_alpha=smooth_frames_alpha, smooth_coordinates_alpha=smooth_coordinates_alpha)
                                    previous_frame = finished_frame
                                    previous_face_center = smoothed_coordinates
                                else:
                                    finished_frame, _ = frame_landscape(frame=frame, clip_shape=(clip_width, clip_height), previous_frame = previous_frame, smooth_frames_alpha=smooth_frames_alpha, smooth_coordinates_alpha=smooth_coordinates_alpha)
                                    previous_frame = finished_frame

                                new_video.append(finished_frame)

                            current_frame_index += 1

                    # Write the processed frames to a new video file
                    output_video = ImageSequenceClip(new_video, fps=fps)
                    output_video.audio = input_video.audio
                    output_video.write_videofile(output_video_path, codec="libx264", audio_codec="aac", temp_audiofile="temp-audio.m4a", remove_temp=True, fps=fps)

        @staticmethod
        def caption(input_video_path: str, output_video_path: str, transcript: dict = None, max_chars_per_line: int = 30, max_words_per_phrase: int = 4, fontsize: int = 40, font: str = 'Komika-Axis', stroke_color: str ='black', stroke_width: int = 1, position: tuple = ('center', 0.6)):
            
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

class Tools:
    """
    Utility class containing various tools and helpers.
    """
    class AI:
        """
        Class for analyzing transcripts for viral content using different AI methods.
        """
        @staticmethod
        def analysis(transcript: str, chunk_size: int = 2000, method: str = "openai", max_amount: int = 3):
            """
            Analyze the transcript for viral content.

            Parameters:
                transcript (str): Formatted transcript.
                chunk_size (int): Size of each chunk to split the transcript into.
                method (str): AI method to use for analysis. Options: "openai", "ollama".
                max_amount (int): Maximum number of viral sections to find.

            Returns:
                list: List of viral content results.
            """
            # Configure the AI method
            config_ai_method = CONFIG_AI.get(method, None)
            api_key = config_ai_method.get("api_key", None)
            if api_key is not None:
                openai.api_key = api_key

            if config_ai_method is None:
                raise Exception("Invalid AI method. Choose from: {}".format(list(CONFIG_AI.keys())))

            # Split the transcript into chunks
            transcript_chunks = [transcript[i:i + chunk_size] for i in range(0, len(transcript), chunk_size)]
            results = []

            # Define methods for different AI platforms
            methods = {
                "openai": lambda messages: openai.ChatCompletion.create(model=config_ai_method.get("model"), messages=messages, max_tokens=config_ai_method.get("max_tokens", None), stop=config_ai_method.get("dtop", None), functions=config_ai_method.get("functions", None)).choices[0].message.function_call.arguments,
                "ollama": lambda messages: ollama.Client(host=config_ai_method.get("host")).chat(model=config_ai_method.get("model"), messages=prompt, format=config_ai_method.get("format", None))['message']['content']
            }
            
            response_template = config_ai_method.get('response_template')
            system_content = config_ai_method.get('system_content')

            for chunk in transcript_chunks:
                prompt = f"This is a transcript of a video. Please identify the most viral section from the whole, must be more than 30 seconds in duration. Make sure you provide extremely accurate timestamps. Here is the Transcription:\n{chunk}"
                messages = [
                    {"role": "system", "content": f"{system_content} {response_template}"},
                    {"role": "user", "content": prompt}
                ]

                try:
                    # Check amount of results.
                    if len(results) >= max_amount:
                            break
                    
                    # Call the AI method to generate a response
                    response = methods.get(method, lambda messages: None)(messages)
                    response_json = json.loads(response)
                    
                    response_json_keys = Tools.JSON.all_keys(response_json)
                    response_template_keys = Tools.JSON.all_keys(response_template)
                    
                    if response_json_keys == response_template_keys:
                        results.append(response_json)

                except ollama.ResponseError as e:
                    print(f"Ollama response error: {e}")

                except openai.error.RateLimitError as e:
                    print(f"Rate limit error: {e}")

                except openai.error.OpenAIError as e:
                    print(f"OpenAI error: {e}")

                except Exception as e:
                    print(f"Error: {e}")

            return results

    class JSON:
        """
        Helper class for JSON-related operations.
        """
        @staticmethod
        def all_keys(dictionary: dict):
            """
            Get all keys of a dictionary.
            """
            keys = []
            for key, value in dictionary.items():
                keys.append(key)
                if isinstance(value, dict):
                    nested_keys = Tools.JSON.all_keys(value)
                    for nested_key in nested_keys:
                        keys.append(key + '.' + nested_key)
            return keys


def main():
    start_time = time.perf_counter()

    youtube_id = "aMcjxSThD54"
    youtube_url = f"https://www.youtube.com/watch?v={youtube_id}"

    # Get video title
    video_info = youtube_dl.YoutubeDL({}).extract_info(youtube_url, download=False)
    video_title = video_info.get("title", None)
    input_video_path = f"{video_title}.mp4"

    # Download the video
    if not os.path.isfile(input_video_path):
        Video.Download.youtube(youtube_url)

    # Extract segment information
    transcript_json = Video.Transcript.get_youtube(youtube_id)
    transcript = Video.Transcript.get_text(transcript_json)

    # Get the viral segments
    segments = Tools.AI.analysis(transcript)

    for segment in segments:
        # Get viral segment
        viral_segment = segment.get("viral", None)

        # Segment data
        segment_start_time = viral_segment.get("start_time", None)
        segment_end_time = viral_segment.get("end_time", None)
        segment_title = viral_segment.get("title", None)

        # Crop the video segment
        input_segment_path, output_segment_path = input_video_path, f"{segment_title}_segment.mp4"
        Video.Edit.crop(input_segment_path, output_segment_path, segment_start_time, segment_end_time)

        # Clip the cropped segment
        input_clip_path, output_clip_path = output_segment_path, f"{segment_title}_clip.mp4"
        Video.Edit.clip(input_clip_path, output_clip_path)
        
    end_time = time.perf_counter()
    print(f"Total time: {end_time - start_time} seconds!")

if __name__ == "__main__":
    main()