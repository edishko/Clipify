from pytube import YouTube
from pytube.exceptions import RegexMatchError, VideoUnavailable
from youtube_transcript_api import YouTubeTranscriptApi
from openai import ChatCompletion
from moviepy.video.io.VideoFileClip import VideoFileClip
import cv2
import json
import os
import time
import face_recognition
from moviepy.editor import VideoFileClip, ImageSequenceClip
import openai

openai.api_key = ""

''' Functions for downloading videos '''

def download_video(url, filename):
    try:
        # Create a YouTube object
        yt = YouTube(url)

        # Get the stream with the highest resolution
        video_stream = yt.streams.get_highest_resolution()

        if video_stream:
            # Download the video
            video_stream.download(filename=filename)
            print("Downloaded video successfully!")
        else:
            print("Error: No video stream found.")
        
    except RegexMatchError as e:
        print(f"Error: {e}")
    except VideoUnavailable as e:
        print(f"Error: Video is unavailable. {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

''' Functions for segmenting and editing videos '''

def segment_video(output_path: str, json_path: str = None, json: dict = None):
    """
    Segment a video based on information from a JSON file.

    Parameters:
    - json_path (str): Path to the JSON file containing video segment information.
    - input_path (str): Path to the input video file.
    - output_path (str): Path to save the segmented video.
    """
    if json_path is None and json is None:
        raise ValueError("Either json_path or json must be provided.")

    elif json_path:
        with open(json_path, 'r') as json_file:
            segment_info = json.load(json_file)
    
    elif json:
        segment_info = json
    
    try:
        # Extract information
        input_path = segment_info.get('input_path', 'input_video.mp4')
        start_time = segment_info.get('start_time', 0.0)
        end_time = segment_info.get('end_time', 0.0)

        # Load the video clip
        video_clip = VideoFileClip(input_path)

        # Set the duration of the segment
        segment_clip = video_clip.subclip(start_time, end_time)

        # Write the segmented video to the output path with highest quality settings
        segment_clip.write_videofile(output_path, codec="libx264", audio_codec="aac", bitrate="5000k")

        # Close the video clips
        video_clip.close()
        segment_clip.close()

        print(f"Segmentation complete. Video saved to {output_path}")

    except Exception as e:
        print(f"Error: {e}")

def smooth_coordinates(current_coordinates: (int, int), previous_coordinates: (int, int) = None, alpha: float = 0.2):
    if previous_coordinates is None:
        return current_coordinates
    else:
        smoothed_x = int(alpha * current_coordinates[0] + (1 - alpha) * previous_coordinates[0])
        smoothed_y = int(alpha * current_coordinates[1] + (1 - alpha) * previous_coordinates[1])
        return smoothed_x, smoothed_y

def clipify(input_video_path: str, output_video_path: str, face_check_interval: int = 3):
    
    input_video = VideoFileClip(input_video_path) # Load the video clip

    # Get the video dimensions and fps
    video_width, video_height = input_video.size
    fps = input_video.fps
    duration = input_video.duration

    output_width, output_height  = 720, 1280 # Adjust based on your desired output resolution
    
    aspect_ratio = 9 / 16

    clip_width, clip_height = int(min(video_width, video_height * aspect_ratio)), int(min(video_height, video_width / aspect_ratio))
    crop_x, crop_y = clip_width // 2, clip_height // 2

    face_center = None # Initialize variables
    video_frames = [] # Initialize a list to store video frames
    
    for index, frame in enumerate(input_video.iter_frames(fps=fps, dtype="uint8")):
        
        if index % face_check_interval == 0:
            
            face_locations = face_recognition.face_locations(img=frame)
            
            if face_locations:
                
                top, right, bottom, left = face_locations[0]
                current_face_center = ((left + right) // 2, (top + bottom) // 2)
                
                previous_face_center = face_center
                face_center = smooth_coordinates(current_face_center, previous_face_center)

                crop_x = max(0, face_center[0] - clip_width // 2)
                crop_y = max(0, face_center[1] - clip_height // 2)

        cropped_frame = frame[crop_y:crop_y + clip_height, crop_x:crop_x + clip_width]
        resized_frame = cv2.resize(cropped_frame, (output_width, output_height))
        finished_frame = resized_frame

        video_frames.append(finished_frame)
            
    output_video = ImageSequenceClip(video_frames, fps=fps) # Create an ImageSequenceClip without specifying the size
    
    # Set audio duration and write the video file
    output_video.audio = input_video.audio.subclip(0, duration)
    output_video = output_video.subclip(0, duration)
    output_video.write_videofile(output_video_path, codec = "libx264", audio_codec = "aac", temp_audiofile = "temp-audio.m4a", remove_temp = True, fps = fps)

''' Functions for retrieving and analyzing transcripts '''

def get_transcript(video_id: str) -> str:
    """
    Get transcript with YoutubeTranscriptAPI.

    Parameters:
    - video_id (str): YouTube video ID.

    Returns:
    - str: Formatted transcript.
    """
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        formatted_transcript = ''

        for entry in transcript:
            start_time = "{:.2f}".format(entry['start'])
            end_time = "{:.2f}".format(entry['start'] + entry['duration'])
            text = entry['text']
            formatted_transcript += f"{start_time} --> {end_time} : {text}\n"

        print('Transcript successfully retrieved!')
        return formatted_transcript

    except Exception as e:
        print(f"Error fetching transcript: {e}")
        return None

def analyze_transcript(transcript: str, save: bool = True, chunk_size = 2000, max_amount: int = 1) -> list:
    """
    Analyze the transcript for viral content.

    Parameters:
    - transcript (str): Formatted transcript.
    - save (bool): Save results to files.
    - max_amount (int): Maximum number of viral sections to find.

    Returns:
    - list: List of viral content results.
    """
    amount = 0
    transcript_chunks = [transcript[i:i + chunk_size] for i in range(0, len(transcript), chunk_size)]
    response_obj_template = '"viral": {"title": "Title here", "start_time": 97.19, "end_time": 127.43}'

    results = []

    for chunk in transcript_chunks:
        prompt = f"This is a transcript of a video. Please identify the most viral section from the whole, must be more than 30 seconds in duration. Make sure you provide extremely accurate timestamps. Here is the Transcription:\n{chunk}"
        messages = [
            {"role": "system", "content": f"You are a ViralGPT helpful assistant. You are a master at reading YouTube transcripts and identifying the most interesting and viral content. You return the most viral moment in this format: {response_obj_template} Returned JSON framework must be constructed with double-quotes. Double quotes within strings must be escaped with backslash, single quotes within strings will not be escaped."},
            {"role": "user", "content": prompt}
        ]

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-1106",
                messages=messages,
                max_tokens=1000,
                n=1,
                stop=None,
                functions=[
                    {
                        "name": "viralsection",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "viral": {
                                    "title": "string",
                                    "start_time": "float",
                                    "end_time": "float",
                                },
                            }
                        }
                    }
                ]
            )
            temp_result = response.choices[0].message.function_call.arguments
            result = None

            try:
                if temp_result:
                    temp_json = json.loads(temp_result)
                    if temp_json:
                        viral = temp_json['viral']
                        print(viral)
                        duration = float(viral['end_time']) - float(viral['start_time'])

                        if 30 <= duration <= 60:
                            print(f"Viral section found in {duration} seconds! Title: {viral['title']}")
                            result = viral
                            results.append(result)
                            amount += 1

                            output_dir = os.path.join('videos', viral['title'])
                            os.makedirs(output_dir, exist_ok=True)

                            if save:
                                with open(os.path.join(output_dir, f"{viral['title']}.json"), 'w') as f:
                                    json.dump(result, f)

                            if amount == max_amount:
                                return results

            except Exception as e:
                print(f"Error parsing JSON: {e}")
                temp_json = None

        except openai.error.RateLimitError as e:
            print(f"Rate limit error: {e}")
        except openai.error.OpenAIError as e:
            print(f"OpenAI error: {e}")

    if results is None:
        analyze_transcript(transcript, save, chunk_size, max_amount)

''' Main function and execution '''

def main():
    start_time = time.time()  # Record the start time

    # https://www.youtube.com/watch?v=DZu3VvmaX9E
    video_id = 'DZu3VvmaX9E'
    url = 'https://www.youtube.com/watch?v=' + video_id
    filename = 'input_video.mp4'
    download_video(url, filename)

    transcript = get_transcript(video_id=video_id)
    if transcript is None:
        main()

    important_segments = analyze_transcript(transcript = transcript, chunk_size = 1000 )

    for segment in important_segments:
        title = f"{segment['title']}"
        
        temp_video_path = f'videos/{title}/{title}_temp0.mp4'
        video_path = f'videos/{title}/{title}.mp4'

        segment_video(json = segment, output_path = temp_video_path)
        time.sleep(1)
        clipify(input_video_path = temp_video_path, output_video_path = video_path)

        # generate_subtitle(input_file = video_path, output_folder = title)
    
    end_time = time.time()  # Record the end time
    print(f"Total time taken: {end_time - start_time} seconds")
    

if __name__ == "__main__":
    main()
