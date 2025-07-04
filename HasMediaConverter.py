from typing import List, Dict, Optional, Any
import os
import subprocess
import argparse
import json
import re
import time


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Transcode videos to DASH or HLS format with optional NVIDIA GPU acceleration. \n"
                    "Output files (MPD and M4S segments for DASH, or M3U8 and TS files for HLS) will be placed in a subdirectory named "
                    "after the input video file (without extension), located in the same directory "
                    "as the input video, or in a specified output base directory."
    )
    parser.add_argument("input_files", nargs="+", help="List of input video file paths.")
    parser.add_argument(
        "-o", "--output-basedir", type=str, default=None,
        help="Optional. Base directory where output subdirectories for each video will be created. "
             "If not specified, output subdirectories are created next to each input video."
    )
    parser.add_argument(
        "--gpu", action="store_true",
        help="Enable NVIDIA GPU acceleration (NVENC H.264 encoding). Requires compatible hardware, drivers, and FFmpeg build."
    )
    parser.add_argument(
        "--format", choices=["dash", "hls"], default="hls",
        help="Specify the output format: 'dash' or 'hls'. Default is 'hls'."
    )
    return parser.parse_args()


def check_command_exists(command: str) -> bool:
    """检查命令是否存在于系统PATH中"""
    try:
        subprocess.run([command, "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def get_video_metadata(video_path: str) -> Optional[Dict[str, Any]]:
    """使用ffprobe获取视频元数据 (分辨率, 时长, 编码等)"""
    if not os.path.isfile(video_path):
        print(f"Error: Input video file not found: {video_path}")
        return None

    ffprobe_cmd = [
        "ffprobe",
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        video_path
    ]
    try:
        result = subprocess.run(ffprobe_cmd, capture_output=True, text=True, check=True, encoding='utf-8')
        metadata = json.loads(result.stdout)

        video_stream = next((s for s in metadata.get("streams", []) if s.get("codec_type") == "video"), None)
        audio_stream = next((s for s in metadata.get("streams", []) if s.get("codec_type") == "audio"), None)

        if not video_stream:
            print(f"Error: No video stream found in {video_path}")
            return None

        duration_str = metadata.get("format", {}).get("duration")
        duration = float(duration_str) if duration_str else 0.0

        return {
            "width": int(video_stream.get("width", 0)),
            "height": int(video_stream.get("height", 0)),
            "duration": duration,
            "codec_name": video_stream.get("codec_name"),
            "has_audio": audio_stream is not None
        }
    except subprocess.CalledProcessError as e:
        print(f"Error running ffprobe for {video_path}: {e.stderr}")
    except json.JSONDecodeError:
        print(f"Error decoding ffprobe JSON output for {video_path}.")
    except Exception as e:
        print(f"An unexpected error occurred while getting metadata for {video_path}: {e}")
    return None


def run_ffmpeg_with_progress(ffmpeg_cmd: List[str], duration: float, output_dir: str) -> bool:
    """执行FFmpeg命令并显示进度"""
    print()
    print(f"Target output directory (cwd for ffmpeg): {output_dir}")
    print(f"Executing FFmpeg command: {' '.join(ffmpeg_cmd)}\n")

    # 使用 Popen 以非阻塞方式读取 stderr
    process = subprocess.Popen(
        ffmpeg_cmd,
        stdout=subprocess.PIPE,  # 可以捕获 stdout，但通常 FFmpeg 主要信息在 stderr
        stderr=subprocess.PIPE,
        text=True,
        encoding='utf-8',
        errors='replace',  # 替换无法解码的字符
        cwd=output_dir  # 关键：设置FFmpeg的工作目录
    )

    # 正则表达式从FFmpeg的stderr中提取时间
    # time=00:00:10.52 bitrate=...
    time_regex = re.compile(r"time=(\d{2}):(\d{2}):(\d{2})\.(\d{2})")

    print("Starting transcoding...")
    while True:
        if process.poll() is not None:  # 进程已结束
            break

        line = process.stderr.readline()
        if not line:  # 没有更多输出了 (可能进程结束了)
            # 短暂等待，确保进程真的结束了
            time.sleep(0.1)
            if process.poll() is not None:
                break
            else:  # 可能是暂时没输出
                continue

        match = time_regex.search(line)
        if match:
            h, m, s, ms = map(int, match.groups())
            elapsed_time = h * 3600 + m * 60 + s + ms / 100.0
            if duration > 0:
                progress = (elapsed_time / duration) * 100
                print(f"\rProgress: {progress:.1f}% ({elapsed_time:.1f}s / {duration:.1f}s)", end="", flush=True)
        else: # 打印其他非进度行，用于调试
            print(line.strip())

    # 等待进程完全结束并获取最终输出
    stdout, stderr = process.communicate()

    if duration > 0:  # 确保最后打印100%
        print(f"\rProgress: 100.0% ({duration:.1f}s / {duration:.1f}s)")
    else:
        print("\rProgress: Done! (Duration was 0 or unknown)")

    if process.returncode == 0:
        print(f"Conversion completed successfully.")
        if stderr and any(line.strip() for line in stderr.splitlines() if
                          "error" not in line.lower() and "warning" not in line.lower()):  # 打印非错误/警告的stderr信息
            # 避免打印所有进度行
            final_stderr_lines = [l for l in stderr.splitlines() if not time_regex.search(l) and l.strip()]
            if final_stderr_lines:
                print("FFmpeg Info (stderr):")
                for l in final_stderr_lines[:10]:  # 只打印前10行
                    print(l)
                if len(final_stderr_lines) > 10:
                    print("...")
        return True
    else:
        print(f"\nError: Conversion failed with exit code {process.returncode}")
        print("FFmpeg stdout:")
        print(stdout.strip())
        print("FFmpeg stderr:")
        print(stderr.strip())
        return False


class MediaConverter:
    def __init__(self):
        # --- 配置常量 ---
        self.OUTPUT_PROFILES: List[Dict[str, Any]] = [
            {"width": 3840, "height": 2160, "cpu_br": "6000k", "gpu_br": "10000k", "name": "2160p"},  # 4K
            {"width": 1920, "height": 1080, "cpu_br": "3000k", "gpu_br": "5000k", "name": "1080p"},
            {"width": 1280, "height": 720, "cpu_br": "1750k", "gpu_br": "2800k", "name": "720p"},
            {"width": 720, "height": 480, "cpu_br": "800k", "gpu_br": "1500k", "name": "480p"},
            # 可以添加更多规格，例如 360p
            # {"width": 640,  "height": 360,  "cpu_br": "500k",  "gpu_br": "800k",   "name": "360p"},
        ]
        self.MIN_OUTPUT_HEIGHT = 480  # 最小输出高度
        self.DEFAULT_AUDIO_BITRATE = "128k"
        self.SEGMENT_DURATION = 2  # 秒
        self.args = parse_arguments()

        """
        1. 检查依赖，包括ffmpeg gpu等
        2. 获取视频元数据
        3. 确定视频的输出规格
        4. 生成 ffmpeg-hls 指令，生成 ffmpeg-dash 指令
        5. 执行指令，并捕获输出信息，打印进度
        """


        print("检查环境依赖 FFmpeg")
        # 1. 检查依赖
        if not check_command_exists("ffmpeg"):
            print("Error: ffmpeg not found in PATH. Please install FFmpeg.")
            return
        if not check_command_exists("ffprobe"):
            print("Error: ffprobe not found in PATH. Please install FFmpeg (it usually includes ffprobe).")
            return



    def determine_target_profiles(self, source_width: int, source_height: int) -> List[Dict[str, Any]]:
        """根据源分辨率确定输出规格列表"""
        use_gpu = self.args.gpu
        target_profiles = []
        for profile in self.OUTPUT_PROFILES:
            if source_height >= profile["height"] and source_width >= profile["width"]:  # 源分辨率大于等于当前profile
                actual_bitrate = profile["gpu_br"] if use_gpu else profile["cpu_br"]
                target_profiles.append({
                    "width": profile["width"],
                    "height": profile["height"],
                    "bitrate": actual_bitrate,
                    "name": profile["name"]
                })

        # 如果没有任何profile符合（例如源视频太小），则至少编码一个不高于源且不低于MIN_OUTPUT_HEIGHT的规格
        if not target_profiles:
            # 尝试使用源分辨率，但确保不低于最小高度
            target_h = max(source_height, self.MIN_OUTPUT_HEIGHT)
            # 按比例计算宽度
            if source_height > 0:
                target_w = int(source_width * (target_h / source_height))
                # 确保宽度是偶数
                target_w = target_w if target_w % 2 == 0 else target_w + 1
            else:  # 无法确定比例
                target_w = max(source_width, int(self.MIN_OUTPUT_HEIGHT * (16 / 9)))  # 假设16:9
                target_w = target_w if target_w % 2 == 0 else target_w + 1

            default_br_key = "gpu_br" if use_gpu else "cpu_br"
            # 尝试找到最接近的较低profile的码率，或最后一个profile的码率
            fallback_bitrate = self.OUTPUT_PROFILES[-1][default_br_key]
            for p in reversed(self.OUTPUT_PROFILES):
                if target_h >= p["height"]:
                    fallback_bitrate = p[default_br_key]
                    break

            target_profiles.append({
                "width": target_w,
                "height": target_h,
                "bitrate": fallback_bitrate,  # 需要一个更智能的码率选择
                "name": f"{target_h}p_fallback"
            })
            print(
                f"Info: Source resolution {source_width}x{source_height} is small. Generating fallback profile: {target_w}x{target_h}@{fallback_bitrate}")

        # 确保至少有一个profile的高度不低于MIN_OUTPUT_HEIGHT，除非源视频本身就低于它
        final_profiles = []
        source_is_smaller_than_min = source_height < self.MIN_OUTPUT_HEIGHT

        for tp in target_profiles:
            if tp["height"] >= self.MIN_OUTPUT_HEIGHT:
                final_profiles.append(tp)
            elif source_is_smaller_than_min and tp["height"] == source_height:  # 如果源就小，允许输出源的规格
                final_profiles.append(tp)
                break  # 通常只输出一个最小的

        if not final_profiles and target_profiles:  # 如果过滤后啥都没了，但之前有，就用之前最后一个
            final_profiles.append(target_profiles[-1])

        # 如果源分辨率小于480p，只输出源分辨率的一个版本
        if source_height < self.MIN_OUTPUT_HEIGHT:
            original_bitrate_key = "gpu_br" if use_gpu else "cpu_br"
            # 尝试找到最接近的较低profile的码率，或最后一个profile的码率
            original_fallback_bitrate = self.OUTPUT_PROFILES[-1][original_bitrate_key]
            for p in reversed(self.OUTPUT_PROFILES):
                if source_height >= p["height"]:  # 可能这个条件永远不会满足
                    original_fallback_bitrate = p[original_bitrate_key]
                    break
            # 如果源太小，使用一个保守的码率
            if source_height < 360: original_fallback_bitrate = "300k" if not use_gpu else "500k"

            final_profiles = [{
                "width": source_width if source_width % 2 == 0 else source_width + 1,
                "height": source_height,
                "bitrate": original_fallback_bitrate,
                "name": f"{source_height}p_source"
            }]
            print(
                f"Info: Source resolution {source_width}x{source_height} is below {self.MIN_OUTPUT_HEIGHT}p. Outputting single source-like profile.")

        # 去重，并按高度降序排序
        unique_profiles_dict = {(p['width'], p['height']): p for p in final_profiles}
        sorted_unique_profiles = sorted(unique_profiles_dict.values(), key=lambda x: x['height'], reverse=True)

        if not sorted_unique_profiles:  # 最终保险
            print(
                f"Warning: No suitable output profiles could be determined for {source_width}x{source_height}. This should not happen.")
            # 返回一个最小的默认值
            default_br_key = "gpu_br" if use_gpu else "cpu_br"
            sorted_unique_profiles.append({
                "width": 640, "height": self.MIN_OUTPUT_HEIGHT, "bitrate": self.OUTPUT_PROFILES[-1][default_br_key],
                "name": f"{self.MIN_OUTPUT_HEIGHT}p_default_fallback"
            })

        return sorted_unique_profiles

    def build_dash_command(self, input_file: str, target_profiles: List[Dict[str, Any]], has_audio: bool) -> List[str]:
        """构建FFmpeg命令列表"""
        use_gpu = self.args.gpu
        cmd: List = ["ffmpeg", "-hide_banner", "-y"]

        cmd.extend(["-i", input_file])

        # 视频流映射和编码参数
        for i, profile in enumerate(target_profiles):
            cmd.extend(["-map", "0:v:0"])  # 映射第一个视频输入流

            w, h, br = profile["width"], profile["height"], profile["bitrate"]


            if use_gpu:
                cmd.extend([f"-c:v:{i}", "h264_nvenc", f"-b:v:{i}", br])
                cmd.extend([f"-preset:v:{i}", "p5"])  # NVENC预设，可调 (p1-p7, default, slow, medium, fast)
                # NVENC 通常需要 maxrate 和 bufsize
                max_br_val = int(br.replace('k', '')) * 2
                buf_br_val = int(br.replace('k', '')) * 4
                cmd.extend([f"-maxrate:v:{i}", f"{max_br_val}k", f"-bufsize:v:{i}", f"{buf_br_val}k"])

                # GPU滤镜链: 上传 -> 缩放(GPU) -> 下载 -> 格式化 -> 填充(CPU) -> 格式化
                # 注意: pad_cuda 可能在某些FFmpeg版本中不可用或行为不同
                # 一个更通用的方法是在下载回CPU后进行填充
                # interp_algo=lanczos 提供高质量缩放
                vf_items = [
                    "hwupload_cuda",
                    f"scale_cuda=w={w}:h={h}:force_original_aspect_ratio=decrease:interp_algo=lanczos",
                    "hwdownload",
                    f"format=nv12",  # NVENC可能偏好nv12
                    f"pad=w={w}:h={h}:x=(ow-iw)/2:y=(oh-ih)/2:color=black",  # CPU填充
                    "format=yuv420p"  # 确保最终为yuv420p
                ]
                cmd.extend([f"-vf:v:{i}", ",".join(vf_items)])

            else:  # CPU
                cmd.extend([f"-c:v:{i}", "libx264", f"-b:v:{i}", br])
                cmd.extend([f"-preset:v:{i}", "medium"])  # libx264预设
                vf_items = [
                    f"scale=w={w}:h={h}:force_original_aspect_ratio=decrease:flags=lanczos",
                    f"pad=w={w}:h={h}:x=(ow-iw)/2:y=(oh-ih)/2:color=black",
                    "format=yuv420p"
                ]
                cmd.extend([f"-vf:v:{i}", ",".join(vf_items)])

            # GOP设置 (对NVENC和libx264都适用)
            gop_size = self.SEGMENT_DURATION * 24  # 假设24fps，可从源视频获取fps
            cmd.extend([f"-g:v:{i}", str(gop_size)])
            cmd.extend([f"-keyint_min:v:{i}", str(gop_size)])  # 确保最小关键帧间隔
            cmd.extend([f"-sc_threshold:v:{i}", "0"])  # 禁用基于场景切换的自动关键帧

        # 音频流映射和编码
        if has_audio:
            cmd.extend(["-map", "0:a:0?"])  # 映射第一个音频输入流 (如果存在)
            cmd.extend(["-c:a:0", "aac", "-b:a:0", self.DEFAULT_AUDIO_BITRATE, "-ac:a:0", "2"])

        # DASH输出参数
        cmd.extend([
            "-use_timeline", "1",
            "-use_template", "1",
            "-seg_duration", str(self.SEGMENT_DURATION),
            "-adaptation_sets", f"id=0,streams=v{' id=1,streams=a' if has_audio else ''}",
            "-f", "dash",
            "-init_seg_name", "init-stream$RepresentationID$.m4s",
            "-media_seg_name", "chunk-stream$RepresentationID$-$Number%05d$.m4s",
            "main.mpd"  # MPD 文件名，将相对于 cwd
        ])
        return cmd


    def build_hls_command(self, input_file: str, target_profiles: List[Dict[str, Any]], has_audio: bool) -> List[str]:
        """构建FFmpeg HLS命令列表"""
        cmd: List = ["ffmpeg", "-hide_banner", "-y"]
        cmd.extend(["-i", input_file])
        cmd.extend(["-filter_complex"])

        # 构建滤镜链
        filter_complex_parts = "[0:v]split={}".format(len(target_profiles))
        filter_complex_parts += "".join("[v{}]".format(1 + i) for i in range(len(target_profiles))) + ";"

        for i, profile in enumerate(target_profiles):
            filter_complex_parts += "[v{}]scale=w={}:h={}[v{}out]".format(i + 1, profile['width'], profile['height'], i + 1)
            if i < len(target_profiles) - 1:
                filter_complex_parts += ";"

        cmd.extend([filter_complex_parts])

        # 视频流映射和编码参数
        for i, profile in enumerate(target_profiles):
            cmd.extend(["-map", f"[v{i + 1}out]"])
            cmd.extend([f"-c:v:{i}", "libx264", f"-b:v:{i}", profile["bitrate"]])
            cmd.extend([f"-maxrate:v:{i}", profile["bitrate"], f"-bufsize:v:{i}", profile["bitrate"]])

        # 音频流映射和编码
        audio_mappings = []
        if has_audio:
            for i in range(len(target_profiles)):
                if i < len(target_profiles) - 1:
                    cmd.extend(["-map", "0:a:0", f"-c:a:{i}", "aac", f"-b:a:{i}", self.DEFAULT_AUDIO_BITRATE])
                else:
                    cmd.extend(["-map", "0:a:0", f"-c:a:{i}", "aac", f"-b:a:{i}", self.DEFAULT_AUDIO_BITRATE, "-ac", "2"])
                audio_mappings.append(f"a:{i}")

        # 动态生成 var_stream_map
        var_stream_map_parts = []
        for i in range(len(target_profiles)):
            video_part = f"v:{i}"
            audio_part = audio_mappings[i] if has_audio else ""
            part = f"{video_part},{audio_part}" if has_audio else video_part
            var_stream_map_parts.append(part)

        var_stream_map = " ".join(var_stream_map_parts)

        # 正确组织 HLS 输出参数
        cmd.extend([
            "-f", "hls",
            "-hls_time", "10",  # 每个 TS 片段的时长（秒）
            "-hls_playlist_type", "vod",  # 播放列表类型为点播
            "-hls_flags", "independent_segments",  # 每个 TS 片段独立
            "-hls_segment_type", "mpegts",  # TS 片段格式
            "-master_pl_name", "main.m3u8",  # 主播放列表文件名
            "-hls_segment_filename", "stream_%v/data%03d.ts",  # TS 片段文件名模板
            "-var_stream_map", var_stream_map, "stream_%v/playlist.m3u8"  # 子播放列表文件名模板
        ])
        return cmd

    def process_video(self, video_file: str):
        # 获取视频文件的绝对地址
        abs_video_file = os.path.abspath(video_file)
        if not os.path.isfile(abs_video_file):
            print(f"Error: Input file not found: {abs_video_file}. Skipping.")
            return

        print(f"\n--- Processing: {abs_video_file} ---")

        metadata = get_video_metadata(abs_video_file)
        if not metadata or metadata["width"] == 0 or metadata["height"] == 0:
            print(f"Error: Could not get valid metadata for {abs_video_file}. Skipping.")
            return

        video_spec = {
            "Source": f"{metadata['width']}x{metadata['height']}",
            "Duration": f"{metadata['duration']:.2f}s",
            "Codec": metadata.get("codec_name", "N/A"),
            "Audio": "Yes" if metadata["has_audio"] else "No"
        }
        print(video_spec)

        target_profiles = self.determine_target_profiles(metadata["width"], metadata["height"])

        if not target_profiles:
            print(f"Error: No target profiles determined for {abs_video_file}. Skipping.")
            return

        print("Target profiles:")
        for tp in target_profiles:
            print(f"  - {tp['width']}x{tp['height']} @ {tp['bitrate']} ({tp['name']})")

        # 确定输出目录
        video_filename_no_ext = os.path.splitext(os.path.basename(abs_video_file))[0]
        if self.args.output_basedir:
            abs_output_basedir = os.path.abspath(self.args.output_basedir)
            output_dir_for_video = os.path.join(abs_output_basedir, video_filename_no_ext)
        else:
            input_file_dir = os.path.dirname(abs_video_file)
            output_dir_for_video = os.path.join(input_file_dir, video_filename_no_ext)

        try:
            os.makedirs(output_dir_for_video, exist_ok=True)
            has_dir = None
            if self.args.format == "dash":
                has_dir = os.path.join(output_dir_for_video, "dash")
            elif self.args.format == "hls":
                has_dir = os.path.join(output_dir_for_video, "hls")
            else:
                print(f"Error: Invalid format specified: {self.args.format}. Skipping.")
                return
            os.makedirs(has_dir, exist_ok=True)
        except OSError as e:
            print(f"Error: Could not create output directory {output_dir_for_video}: {e}. Skipping.")
            return

        if self.args.format == "dash":
            ffmpeg_command = self.build_dash_command(
                abs_video_file,
                target_profiles,
                metadata["has_audio"]
            )
            success_msg = f"Successfully created DASH content for {abs_video_file} in {os.path.join(output_dir_for_video, 'dash')}"
        else:
            ffmpeg_command = self.build_hls_command(
                abs_video_file,
                target_profiles,
                metadata["has_audio"]
            )
            success_msg = f"Successfully created HLS content for {abs_video_file} in {os.path.join(output_dir_for_video, 'hls')}"

        print("FFmpeg command: ", ffmpeg_command)

        success = run_ffmpeg_with_progress(ffmpeg_command, metadata["duration"], has_dir)
        if success:
            print(success_msg)
        else:
            print(f"Failed to create {self.args.format.upper()} content for {abs_video_file}")

    def media2has(self):
        if self.args.gpu:
            """
            检查是否有NVENC
            ffmpeg -codecs 
            
            """
            codecs_result = (subprocess.run(["ffmpeg", "-codecs"], capture_output=True, text=True, check=True, encoding='utf-8'))
            if "h264_nvenc" not in codecs_result.stdout and "hevc_nvenc" not in codecs_result.stdout:
                print("Warning: --gpu specified, but 'h264_nvenc' or 'hevc_nvenc' not found in `ffmpeg -codecs` output. GPU acceleration might not work or might fallback to CPU.")
            else:
                print("Info: GPU acceleration requested. Will attempt to use NVENC.")

        for input_file_arg in self.args.input_files:
            self.process_video(input_file_arg)

if __name__ == "__main__":
    converter = MediaConverter()
    converter.media2has()

