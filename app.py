from flask import Flask, request, jsonify, send_file, render_template
import os
import requests
from moviepy.video.io.VideoFileClip import VideoFileClip
import uuid
import threading
from werkzeug.utils import secure_filename
import csv
import subprocess
import shutil
import time
import gc
import re

# thư viện chuyển đổi audio thành văn bản
import speech_recognition as sr
from pydub import AudioSegment

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'temp'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max

# Tạo thư mục temp nếu chưa có
os.makedirs('temp', exist_ok=True)

# Cache cho PhoWhisper model (chỉ load 1 lần)
_phowhisper_model = None
_phowhisper_pipe = None

def post_process_text(text):
    """Sửa lỗi thường gặp trong text đã transcribe"""
    if not text:
        return text
    
    # Dictionary để sửa lỗi thường gặp
    corrections = {
        # Tên thương hiệu
        r'\bHADEN SHOW\s*ĐỂS?\b': 'Head & Shoulders',
        r'\bHADEN\s*SHOW\s*ĐỂS?\b': 'Head & Shoulders',
        r'\bHADEN\s*SHOW\s*ĐỂ\b': 'Head & Shoulders',
        r'\bHADEN\s*SHOW\s*ĐỂ\s*S\b': 'Head & Shoulders',
        r'\bHADEN\s*SHOW\s*ĐỂ\s*S\s*Mỳ\b': 'Head & Shoulders. Mỳ',
        r'\bHADEN\s*SHOWder\b': 'Head & Shoulders',
        r'\bHADEN\s*SHOWer\b': 'Head & Shoulders',
        r'\bPUT\s*CHO\s*CÊ\b': 'Cô cho con',
        r'\bCHO\s*CÊ\b': 'cho con',
        r'\bCÊ\b': 'con',
        # Sửa dấu câu
        r'\s+([,\.!?;:])': r'\1',  # Bỏ space trước dấu câu
        r'([,\.!?;:])\s*([A-ZĐ])': r'\1 \2',  # Thêm space sau dấu câu
        # Sửa lỗi thường gặp
        r'\bkhông\s+có\s+gầu\s+vì\s+luôn\s+có\b': 'không có gầu vì luôn có Head & Shoulders',
    }
    
    for pattern, replacement in corrections.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    # Chuẩn hóa khoảng trắng
    text = re.sub(r'\s+', ' ', text)  # Nhiều space thành 1 space
    text = text.strip()
    
    return text

def setup_ffmpeg():
    """Thiết lập ffmpeg từ imageio_ffmpeg cho Whisper và pydub"""
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        ffmpeg_exe = get_ffmpeg_exe()
        
        # Set cho Whisper - cần set trước khi load model
        try:
            import whisper
            # Whisper sử dụng subprocess để gọi ffmpeg, cần set trong PATH hoặc dùng biến môi trường
            import os as os_module
            ffmpeg_dir = os.path.dirname(ffmpeg_exe)
            if ffmpeg_dir not in os_module.environ.get("PATH", ""):
                os_module.environ["PATH"] = ffmpeg_dir + os.pathsep + os_module.environ.get("PATH", "")
            # Cũng có thể set trực tiếp trong whisper.audio
            if hasattr(whisper, 'audio'):
                whisper.audio.ffmpeg_path = ffmpeg_exe
        except:
            pass
        
        # Set cho pydub
        import os as os_module
        ffmpeg_dir = os.path.dirname(ffmpeg_exe)
        if ffmpeg_dir not in os_module.environ.get("PATH", ""):
            os_module.environ["PATH"] = ffmpeg_dir + os.pathsep + os_module.environ.get("PATH", "")
        AudioSegment.converter = ffmpeg_exe
        # Tìm ffprobe
        ffprobe_exe = ffmpeg_exe.replace('ffmpeg', 'ffprobe')
        if not os.path.exists(ffprobe_exe):
            # Thử tìm trong cùng thư mục
            ffprobe_exe = os.path.join(ffmpeg_dir, 'ffprobe.exe')
        if os.path.exists(ffprobe_exe):
            AudioSegment.ffprobe = ffprobe_exe
        else:
            AudioSegment.ffprobe = ffmpeg_exe  # Fallback
        
        print(f"Đã thiết lập ffmpeg: {ffmpeg_exe}")
        print(f"FFprobe: {AudioSegment.ffprobe}")
        return True
    except Exception as e:
        print(f"Không thể thiết lập ffmpeg từ imageio_ffmpeg: {e}")
        import traceback
        traceback.print_exc()
        return False

def transcribe_audio(audio_path):
    """Chuyển audio thành văn bản sử dụng Whisper hoặc SpeechRecognition"""
    try:
        print(f"Đang chuyển audio thành văn bản: {audio_path}")
        
        # Kiểm tra file audio có tồn tại không
        if not os.path.exists(audio_path):
            print(f"File audio không tồn tại: {audio_path}")
            return None
        
        file_size = os.path.getsize(audio_path)
        print(f"Kích thước file audio: {file_size} bytes")
        
        # Phương pháp 1: Thử PhoWhisper (tối ưu cho tiếng Việt)
        global _phowhisper_pipe
        try:
            from transformers import pipeline
            import torch
            
            print("Sử dụng PhoWhisper để chuyển đổi (tối ưu cho tiếng Việt)...")
            
            # Thiết lập ffmpeg cho transformers pipeline
            from imageio_ffmpeg import get_ffmpeg_exe
            ffmpeg_exe = get_ffmpeg_exe()
            import os as os_module
            ffmpeg_dir = os.path.dirname(ffmpeg_exe)
            if ffmpeg_dir not in os_module.environ.get("PATH", ""):
                os_module.environ["PATH"] = ffmpeg_dir + os.pathsep + os_module.environ.get("PATH", "")
            
            # Sử dụng model đã cache hoặc load mới
            if _phowhisper_pipe is None:
                # Load PhoWhisper model từ Hugging Face
                # Ưu tiên large model để có độ chính xác cao hơn
                try:
                    # Thử model large trước (chính xác hơn, đặc biệt với tên thương hiệu)
                    print("Đang tải PhoWhisper-large model (chính xác hơn, lần đầu tiên có thể mất vài phút)...")
                    _phowhisper_pipe = pipeline(
                        "automatic-speech-recognition",
                        model="vinai/PhoWhisper-large",
                        device=0 if torch.cuda.is_available() else -1,  # GPU nếu có, không thì CPU
                    )
                    print("Đã tải PhoWhisper-large model thành công")
                except Exception as e:
                    print(f"Không thể tải PhoWhisper-large, thử base: {e}")
                    # Fallback sang base model (nhẹ hơn, nhanh hơn)
                    print("Đang tải PhoWhisper-base model (lần đầu tiên có thể mất vài phút)...")
                    _phowhisper_pipe = pipeline(
                        "automatic-speech-recognition",
                        model="vinai/PhoWhisper-base",
                        device=0 if torch.cuda.is_available() else -1,
                    )
                    print("Đã tải PhoWhisper-base model thành công")
            else:
                print("Sử dụng PhoWhisper model đã cache (không cần tải lại)")
            
            # Chuyển đổi audio thành numpy array để tránh vấn đề ffmpeg trong pipeline
            print("Chuyển đổi và cải thiện chất lượng audio cho PhoWhisper...")
            temp_wav_file = None
            try:
                setup_ffmpeg()  # Đảm bảo có ffmpeg
                from imageio_ffmpeg import get_ffmpeg_exe
                ffmpeg_exe = get_ffmpeg_exe()
                
                # Dùng ffmpeg để convert MP3 sang WAV (không cần ffprobe)
                # Sau đó load WAV bằng pydub hoặc numpy
                if audio_path.endswith('.mp3'):
                    temp_wav_file = audio_path.replace('.mp3', '_temp_phowhisper.wav')
                    print(f"Chuyển đổi MP3 sang WAV bằng ffmpeg: {temp_wav_file}")
                    
                    # Dùng ffmpeg để convert: MP3 -> WAV, 16kHz, mono
                    cmd = [
                        ffmpeg_exe,
                        '-i', audio_path,
                        '-ar', '16000',  # Sample rate 16kHz
                        '-ac', '1',      # Mono
                        '-f', 'wav',     # Format WAV
                        '-y',            # Overwrite
                        temp_wav_file
                    ]
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                    if result.returncode != 0:
                        raise Exception(f"FFmpeg error: {result.stderr}")
                    if not os.path.exists(temp_wav_file):
                        raise Exception("FFmpeg không tạo được file WAV")
                    print(f"Đã chuyển đổi sang WAV thành công")
                    
                    # Load WAV bằng pydub (WAV không cần ffprobe)
                    audio = AudioSegment.from_wav(temp_wav_file)
                else:
                    # Thử load file khác trực tiếp
                    try:
                        audio = AudioSegment.from_file(audio_path)
                    except:
                        # Nếu không được, convert bằng ffmpeg
                        temp_wav_file = audio_path.rsplit('.', 1)[0] + '_temp_phowhisper.wav'
                        cmd = [
                            ffmpeg_exe,
                            '-i', audio_path,
                            '-ar', '16000',
                            '-ac', '1',
                            '-f', 'wav',
                            '-y',
                            temp_wav_file
                        ]
                        subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=True)
                        audio = AudioSegment.from_wav(temp_wav_file)
                
                # Cải thiện chất lượng audio
                # 1. Normalize volume (chuẩn hóa âm lượng)
                audio = audio.normalize()
                
                # 2. Đảm bảo 16kHz, mono (đã convert ở trên nhưng kiểm tra lại)
                if audio.frame_rate != 16000:
                    audio = audio.set_frame_rate(16000)
                if audio.channels != 1:
                    audio = audio.set_channels(1)
                
                # 3. Tăng volume nếu quá nhỏ (nhưng không quá lớn)
                if audio.max_possible_amplitude:
                    max_dBFS = audio.max_dBFS
                    if max_dBFS < -20:  # Nếu quá nhỏ
                        audio = audio.apply_gain(-max_dBFS - 20)  # Tăng lên
                
                # Chuyển đổi thành numpy array (float32, -1.0 đến 1.0)
                import numpy as np
                samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
                if audio.channels == 2:
                    samples = samples.reshape((-1, 2)).mean(axis=1)  # Stereo to mono
                samples = samples / (1 << (8 * audio.sample_width - 1))  # Normalize to [-1, 1]
                
                sampling_rate = audio.frame_rate
                print(f"Đã chuyển đổi audio thành numpy array: {len(samples)} samples, {sampling_rate}Hz")
                
                # Đưa numpy array vào pipeline (không cần ffmpeg)
                # Pipeline sẽ tự động detect numpy array và sử dụng trực tiếp
                print("Bắt đầu chuyển đổi audio thành văn bản...")
                result = _phowhisper_pipe(samples, return_timestamps=False)
                text = result["text"].strip()
                
            except Exception as e:
                print(f"Lỗi khi xử lý audio: {e}")
                import traceback
                traceback.print_exc()
                # Fallback: thử dùng file trực tiếp (có thể sẽ lỗi ffmpeg)
                print("Thử dùng file audio trực tiếp...")
                try:
                    result = _phowhisper_pipe(audio_path, return_timestamps=False)
                    text = result["text"].strip()
                except Exception as e2:
                    print(f"Fallback cũng lỗi: {e2}")
                    raise e  # Raise lỗi ban đầu
            finally:
                # Xóa file WAV tạm nếu có
                if temp_wav_file and os.path.exists(temp_wav_file):
                    try:
                        os.remove(temp_wav_file)
                    except:
                        pass
            
            # Post-processing: Sửa lỗi thường gặp
            text = post_process_text(text)
            
            if text:
                print(f"PhoWhisper: Đã chuyển đổi thành công ({len(text)} ký tự)")
                print(f"Văn bản: {text[:100]}...")
                return text
            else:
                print("PhoWhisper: Không tìm thấy văn bản trong audio")
        except ImportError:
            print("PhoWhisper chưa được cài đặt. Chạy: pip install transformers torch")
        except Exception as e:
            print(f"PhoWhisper lỗi: {e}, thử Whisper...")
            import traceback
            traceback.print_exc()
        
        # Phương pháp 2: Thử Whisper (fallback)
        try:
            import whisper
            print("Sử dụng Whisper để chuyển đổi...")
            # Thiết lập ffmpeg cho Whisper
            from imageio_ffmpeg import get_ffmpeg_exe
            ffmpeg_exe = get_ffmpeg_exe()
            
            # Patch hàm load_audio của Whisper để sử dụng ffmpeg từ imageio_ffmpeg
            import whisper.audio as whisper_audio
            original_load_audio = whisper_audio.load_audio
            
            def patched_load_audio(file, sr=16000):
                """Patch load_audio để sử dụng ffmpeg từ imageio_ffmpeg"""
                import subprocess
                cmd = [
                    ffmpeg_exe,
                    "-nostdin",
                    "-threads", "0",
                    "-i", file,
                    "-f", "s16le",
                    "-ac", "1",
                    "-acodec", "pcm_s16le",
                    "-ar", str(sr),
                    "-"
                ]
                try:
                    out = subprocess.run(cmd, capture_output=True, check=True, input=None).stdout
                except subprocess.CalledProcessError as e:
                    raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e
                except FileNotFoundError:
                    raise RuntimeError(f"ffmpeg not found. Please install ffmpeg or use imageio_ffmpeg.")
                
                import numpy as np
                return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
            
            # Thay thế hàm load_audio
            whisper_audio.load_audio = patched_load_audio
            
            model = whisper.load_model("base")  # base, small, medium, large
            print("Model Whisper đã tải xong, bắt đầu chuyển đổi...")
            result = model.transcribe(audio_path, language="vi")  # Tiếng Việt
            text = result["text"].strip()
            if text:
                # Post-processing: Sửa lỗi thường gặp
                text = post_process_text(text)
                print(f"Whisper: Đã chuyển đổi thành công ({len(text)} ký tự)")
                print(f"Văn bản: {text[:100]}...")
                return text
            else:
                print("Whisper: Không tìm thấy văn bản trong audio")
        except ImportError:
            print("Whisper chưa được cài đặt. Chạy: pip install openai-whisper")
        except Exception as e:
            print(f"Whisper lỗi: {e}")
            import traceback
            traceback.print_exc()
        
        # Phương pháp 3: Sử dụng SpeechRecognition với Google Speech API
        try:
            print("Sử dụng SpeechRecognition với Google Speech API...")
            # Chuyển đổi MP3 sang WAV nếu cần (Google Speech API yêu cầu WAV)
            audio_file = audio_path
            wav_path = None
            
            if audio_path.endswith('.mp3'):
                wav_path = audio_path.replace('.mp3', '.wav')
                print(f"Chuyển đổi MP3 sang WAV: {wav_path}")
                try:
                    # Thiết lập ffmpeg cho pydub
                    setup_ffmpeg()
                    audio = AudioSegment.from_mp3(audio_path)
                    # Đảm bảo format phù hợp với Google Speech API
                    audio = audio.set_frame_rate(16000).set_channels(1)  # Mono, 16kHz
                    audio.export(wav_path, format="wav")
                    audio_file = wav_path
                    print(f"Đã chuyển đổi sang WAV thành công")
                except Exception as e:
                    print(f"Lỗi khi chuyển đổi MP3 sang WAV: {e}")
                    import traceback
                    traceback.print_exc()
                    return None
            
            # Kiểm tra file WAV có tồn tại không
            if not os.path.exists(audio_file):
                print(f"File audio không tồn tại: {audio_file}")
                return None
            
            r = sr.Recognizer()
            print("Đang đọc file audio...")
            with sr.AudioFile(audio_file) as source:
                # Điều chỉnh cho tiếng ồn (thời gian ngắn hơn để nhanh hơn)
                print("Điều chỉnh cho tiếng ồn...")
                r.adjust_for_ambient_noise(source, duration=0.5)
                print("Đang ghi âm...")
                audio_data = r.record(source)
                print(f"Đã ghi âm: {len(audio_data.raw_data)} bytes")
            
            # Thử nhận diện với Google Speech API (miễn phí)
            print("Đang gửi đến Google Speech API...")
            try:
                text = r.recognize_google(audio_data, language="vi-VN")
                # Post-processing: Sửa lỗi thường gặp
                text = post_process_text(text)
                print(f"Google Speech API: Đã chuyển đổi thành công ({len(text)} ký tự)")
                print(f"Văn bản: {text[:100]}...")
                # Xóa file WAV tạm nếu có
                if wav_path and os.path.exists(wav_path):
                    safe_delete_file(wav_path)
                return text
            except sr.UnknownValueError:
                print("Google Speech API không thể nhận diện audio. Có thể audio không có lời nói hoặc chất lượng kém.")
            except sr.RequestError as e:
                print(f"Lỗi khi gọi Google Speech API: {e}")
                print("Có thể cần kết nối internet hoặc API đang bị giới hạn")
            
            # Xóa file WAV tạm nếu có
            if wav_path and os.path.exists(wav_path):
                safe_delete_file(wav_path)
                
        except ImportError:
            print("SpeechRecognition chưa được cài đặt. Chạy: pip install SpeechRecognition pydub")
        except Exception as e:
            print(f"SpeechRecognition lỗi: {e}")
            import traceback
            traceback.print_exc()
        
        print("Tất cả các phương pháp đều thất bại")
        return None
        
    except Exception as e:
        print(f"Lỗi khi chuyển audio thành văn bản: {e}")
        import traceback
        traceback.print_exc()
        return None

def safe_delete_file(file_path, max_retries=5, delay=1):
    """Xóa file an toàn với retry và delay"""
    if not os.path.exists(file_path):
        return True
    
    for attempt in range(max_retries):
        try:
            # Thử xóa file
            os.remove(file_path)
            print(f"Đã xóa file: {file_path}")
            return True
        except PermissionError as e:
            if attempt < max_retries - 1:
                print(f"File đang được sử dụng, đợi {delay} giây và thử lại... (lần {attempt + 1}/{max_retries})")
                time.sleep(delay)
                # Force garbage collection để giải phóng resources
                gc.collect()
            else:
                print(f"Không thể xóa file sau {max_retries} lần thử: {file_path}")
                print(f"Lỗi: {e}")
                return False
        except Exception as e:
            print(f"Lỗi khi xóa file {file_path}: {e}")
            return False
    
    return False

def resolve_url(url):
    """Resolve URL redirect để lấy URL thực tế"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': '*/*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Connection': 'keep-alive',
            'Referer': 'https://www.google.com/'
        }
        # Sử dụng GET với stream=False để follow redirects và lấy URL cuối cùng
        session = requests.Session()
        response = session.get(url, headers=headers, timeout=30, allow_redirects=True, stream=False)
        final_url = response.url
        print(f"URL gốc: {url[:100]}...")
        print(f"URL thực tế: {final_url[:100]}...")
        return final_url
    except Exception as e:
        print(f"Không thể resolve URL, sử dụng URL gốc: {e}")
        return url

def download_video(url, output_path):
    """Tải video từ URL"""
    try:
        # Resolve URL redirect trước
        actual_url = resolve_url(url)
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': '*/*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Referer': 'https://www.google.com/'
        }
        response = requests.get(actual_url, stream=True, timeout=120, headers=headers, allow_redirects=True)
        response.raise_for_status()
        
        # Kiểm tra content-type
        content_type = response.headers.get('content-type', '').lower()
        if 'video' not in content_type and 'application' not in content_type and content_type != '':
            print(f"Cảnh báo: Content-Type không phải video: {content_type}")
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
        
        # Kiểm tra file có được tải về không
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            print(f"Lỗi: File video không được tải về hoặc file rỗng")
            return False
        
        print(f"Đã tải video thành công: {os.path.getsize(output_path)} bytes")
        return True
    except requests.exceptions.Timeout:
        print(f"Lỗi: Timeout khi tải video")
        return False
    except requests.exceptions.HTTPError as e:
        print(f"Lỗi HTTP {e.response.status_code}: {e}")
        return False
    except requests.exceptions.RequestException as e:
        print(f"Lỗi khi tải video: {e}")
        return False
    except Exception as e:
        print(f"Lỗi không xác định khi tải video: {e}")
        return False

def check_audio_track(video_path):
    """Kiểm tra video có audio track không bằng ffprobe"""
    try:
        # Kiểm tra ffprobe có sẵn không
        ffprobe_path = shutil.which('ffprobe')
        if not ffprobe_path:
            # Thử tìm trong imageio_ffmpeg
            try:
                from imageio_ffmpeg import get_ffmpeg_exe
                ffmpeg_exe = get_ffmpeg_exe()
                ffprobe_path = ffmpeg_exe.replace('ffmpeg', 'ffprobe')
                if not os.path.exists(ffprobe_path):
                    return None  # Không thể kiểm tra
            except:
                return None
        
        cmd = [
            ffprobe_path,
            '-v', 'error',
            '-select_streams', 'a:0',
            '-show_entries', 'stream=codec_type',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        has_audio = result.returncode == 0 and 'audio' in result.stdout.lower()
        return has_audio
    except Exception as e:
        print(f"Không thể kiểm tra audio track: {e}")
        return None

def extract_audio_ffmpeg(video_path, audio_path):
    """Trích xuất audio bằng ffmpeg trực tiếp"""
    try:
        # Tìm ffmpeg
        ffmpeg_path = shutil.which('ffmpeg')
        if not ffmpeg_path:
            # Thử tìm trong imageio_ffmpeg
            try:
                from imageio_ffmpeg import get_ffmpeg_exe
                ffmpeg_path = get_ffmpeg_exe()
            except Exception as e:
                print(f"Không tìm thấy ffmpeg: {e}")
                return False
        
        print(f"Sử dụng ffmpeg trực tiếp: {ffmpeg_path}")
        
        cmd = [
            ffmpeg_path,
            '-i', video_path,
            '-vn',  # Không copy video
            '-acodec', 'libmp3lame',  # Sử dụng MP3 codec
            '-ab', '192k',  # Bitrate
            '-ar', '44100',  # Sample rate
            '-y',  # Overwrite output file
            audio_path
        ]
        
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=300  # 5 phút timeout
        )
        
        if result.returncode != 0:
            print(f"FFmpeg error: {result.stderr}")
            return False
        
        if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
            print(f"Đã trích xuất audio bằng ffmpeg thành công: {os.path.getsize(audio_path)} bytes")
            return True
        else:
            print("FFmpeg không tạo được file audio")
            return False
            
    except subprocess.TimeoutExpired:
        print("FFmpeg timeout")
        return False
    except Exception as e:
        print(f"Lỗi khi sử dụng ffmpeg: {e}")
        import traceback
        traceback.print_exc()
        return False

def extract_audio(video_path, audio_path):
    """Trích xuất audio từ video - thử nhiều phương pháp"""
    try:
        # Kiểm tra file video có tồn tại và hợp lệ không
        if not os.path.exists(video_path):
            print(f"Lỗi: File video không tồn tại: {video_path}")
            return False
        
        file_size = os.path.getsize(video_path)
        if file_size == 0:
            print(f"Lỗi: File video rỗng")
            return False
        
        print(f"Đang trích xuất audio từ: {video_path} ({file_size} bytes)")
        
        # Kiểm tra audio track trước
        has_audio = check_audio_track(video_path)
        if has_audio is False:
            print("Video không có audio track")
            return False
        elif has_audio is None:
            print("Không thể kiểm tra audio track, tiếp tục thử...")
        
        # Phương pháp 1: Thử MoviePy
        video = None
        audio = None
        try:
            print("Thử phương pháp 1: MoviePy...")
            video = VideoFileClip(video_path)
            
            if video.audio is None:
                print("MoviePy: Video không có audio track")
                if video:
                    video.close()
                    video = None
                # Đợi một chút để đảm bảo resource được giải phóng
                time.sleep(0.5)
                gc.collect()
                # Thử ffmpeg trực tiếp
                return extract_audio_ffmpeg(video_path, audio_path)
            
            audio = video.audio
            audio.write_audiofile(
                audio_path, 
                verbose=False, 
                logger=None, 
                codec='mp3',
                bitrate='192k'
            )
            
            # Đóng resources
            if audio:
                audio.close()
                audio = None
            if video:
                video.close()
                video = None
            
            # Đợi một chút để đảm bảo resource được giải phóng
            time.sleep(0.5)
            gc.collect()
            
            # Kiểm tra file audio đã được tạo chưa
            if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
                print(f"Đã trích xuất audio bằng MoviePy thành công: {os.path.getsize(audio_path)} bytes")
                return True
            else:
                print("MoviePy không tạo được file, thử ffmpeg...")
                return extract_audio_ffmpeg(video_path, audio_path)
                
        except Exception as e:
            print(f"MoviePy thất bại: {e}")
            # Đảm bảo đóng resources
            try:
                if audio:
                    audio.close()
                if video:
                    video.close()
            except:
                pass
            time.sleep(0.5)
            gc.collect()
            print("Thử phương pháp 2: FFmpeg trực tiếp...")
            return extract_audio_ffmpeg(video_path, audio_path)
        
    except Exception as e:
        print(f"Lỗi khi trích xuất audio: {e}")
        import traceback
        traceback.print_exc()
        # Thử ffmpeg như phương án cuối cùng
        print("Thử phương pháp cuối cùng: FFmpeg...")
        return extract_audio_ffmpeg(video_path, audio_path)

@app.route('/')
def index():
    """Trang chủ"""
    return render_template('index.html')

@app.route('/api/urls', methods=['GET'])
def get_urls():
    """Lấy danh sách URLs từ CSV"""
    try:
        csv_file = "FINAL_RESULT_chunk_0_90000.csv"
        urls = []
        
        if not os.path.exists(csv_file):
            return jsonify({'error': 'File CSV không tồn tại'}), 404
        
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                url = row.get('decoded_url', '').strip().strip('"')
                if url:
                    urls.append({
                        'id': idx + 1,
                        'url': url,
                        'preview': url[:80] + '...' if len(url) > 80 else url
                    })
        
        return jsonify({'urls': urls, 'total': len(urls)})
    except Exception as e:
        return jsonify({'error': f'Lỗi khi đọc CSV: {str(e)}'}), 500

@app.route('/convert', methods=['POST'])
def convert_video():
    """API endpoint để chuyển đổi video thành audio"""
    try:
        data = request.get_json()
        video_url = data.get('url', '').strip()
        
        if not video_url:
            return jsonify({'error': 'Vui lòng nhập URL video'}), 400
        
        # Tạo ID duy nhất cho request
        request_id = str(uuid.uuid4())
        video_filename = os.path.join(app.config['UPLOAD_FOLDER'], f'video_{request_id}.mp4')
        audio_filename = os.path.join(app.config['UPLOAD_FOLDER'], f'audio_{request_id}.mp3')
        
        # Tải video
        download_result = download_video(video_url, video_filename)
        if not download_result:
            error_msg = 'Không thể tải video từ URL này. Có thể URL đã hết hạn hoặc không hợp lệ.'
            return jsonify({'error': error_msg}), 400
        
        # Trích xuất audio
        extract_result = extract_audio(video_filename, audio_filename)
        if not extract_result:
            # Kiểm tra xem video có audio track không
            has_audio = check_audio_track(video_filename)
            if has_audio is False:
                error_msg = 'Video này không có audio track. Vui lòng thử video khác.'
            elif has_audio is None:
                error_msg = 'Không thể trích xuất audio từ video. Có thể video không có audio track, định dạng không được hỗ trợ, hoặc file video bị lỗi. Vui lòng kiểm tra lại file video.'
            else:
                error_msg = 'Không thể trích xuất audio từ video. Có thể do định dạng không được hỗ trợ hoặc file video bị lỗi.'
            
            # Xóa file video nếu có lỗi
            safe_delete_file(video_filename)
            return jsonify({'error': error_msg}), 500
        
        # Xóa file video tạm sau khi trích xuất thành công
        # Đợi một chút để đảm bảo tất cả resources được giải phóng
        time.sleep(1)
        gc.collect()
        safe_delete_file(video_filename)
        
        # Chuyển audio thành văn bản
        statusText = "Đang chuyển audio thành văn bản..."
        text = transcribe_audio(audio_filename)
        
        if text:
            # Xóa file audio sau khi đã lấy được văn bản
            time.sleep(0.5)
            gc.collect()
            safe_delete_file(audio_filename)
            print(f"Đã xóa file audio sau khi lấy được văn bản: {audio_filename}")
            
            # Trả về văn bản (không có download_url vì đã xóa file)
            return jsonify({
                'success': True,
                'text': text,
                'download_url': None,
                'filename': None
            })
        else:
            # Kiểm tra xem có thư viện cần thiết không
            missing_libs = []
            try:
                from transformers import pipeline
                import torch
            except ImportError:
                missing_libs.append('transformers torch')
            
            try:
                import whisper
            except ImportError:
                missing_libs.append('openai-whisper')
            
            try:
                import speech_recognition
            except ImportError:
                missing_libs.append('SpeechRecognition')
            
            try:
                from pydub import AudioSegment
            except ImportError:
                missing_libs.append('pydub')
            
            error_msg = 'Không thể chuyển audio thành văn bản.'
            if missing_libs:
                error_msg += f' Cần cài đặt: pip install {" ".join(missing_libs)}'
            else:
                error_msg += ' Có thể audio không có lời nói, chất lượng audio kém, hoặc cần kết nối internet (cho Google Speech API).'
            
            # Nếu không chuyển được thành text, giữ lại file audio để người dùng có thể tải về
            return jsonify({
                'success': True,
                'text': None,
                'error': error_msg,
                'download_url': f'/download/{request_id}',
                'filename': f'audio_{request_id}.mp3'
            })
        
    except Exception as e:
        return jsonify({'error': f'Lỗi: {str(e)}'}), 500

@app.route('/download/<request_id>')
def download_audio(request_id):
    """Tải file audio"""
    audio_filename = os.path.join(app.config['UPLOAD_FOLDER'], f'audio_{request_id}.mp3')
    
    if not os.path.exists(audio_filename):
        return jsonify({'error': 'File không tồn tại'}), 404
    
    return send_file(
        audio_filename,
        as_attachment=True,
        download_name=f'audio_{request_id}.mp3',
        mimetype='audio/mpeg'
    )

@app.route('/cleanup/<request_id>', methods=['DELETE'])
def cleanup(request_id):
    """Xóa file audio sau khi tải xong"""
    audio_filename = os.path.join(app.config['UPLOAD_FOLDER'], f'audio_{request_id}.mp3')
    
    if os.path.exists(audio_filename):
        try:
            os.remove(audio_filename)
            return jsonify({'success': True})
        except:
            return jsonify({'error': 'Không thể xóa file'}), 500
    
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

