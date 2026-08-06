import math, warnings
import torch, torchvision, imageio, os
import imageio.v3 as iio
from PIL import Image


class DataProcessingPipeline:
    def __init__(self, operators=None):
        self.operators: list[DataProcessingOperator] = [] if operators is None else operators
        
    def __call__(self, data):
        for operator in self.operators:
            data = operator(data)
        return data
    
    def __rshift__(self, pipe):
        if isinstance(pipe, DataProcessingOperator):
            pipe = DataProcessingPipeline([pipe])
        return DataProcessingPipeline(self.operators + pipe.operators)


class DataProcessingOperator:
    def __call__(self, data):
        raise NotImplementedError("DataProcessingOperator cannot be called directly.")
    
    def __rshift__(self, pipe):
        if isinstance(pipe, DataProcessingOperator):
            pipe = DataProcessingPipeline([pipe])
        return DataProcessingPipeline([self]).__rshift__(pipe)


class DataProcessingOperatorRaw(DataProcessingOperator):
    def __call__(self, data):
        return data


class ToInt(DataProcessingOperator):
    def __call__(self, data):
        return int(data)


class ToFloat(DataProcessingOperator):
    def __call__(self, data):
        return float(data)


class ToStr(DataProcessingOperator):
    def __init__(self, none_value=""):
        self.none_value = none_value
    
    def __call__(self, data):
        if data is None: data = self.none_value
        return str(data)


class LoadImage(DataProcessingOperator):
    def __init__(self, convert_RGB=True, convert_RGBA=False):
        self.convert_RGB = convert_RGB
        self.convert_RGBA = convert_RGBA
    
    def __call__(self, data: str):
        image = Image.open(data)
        if self.convert_RGB: image = image.convert("RGB")
        if self.convert_RGBA: image = image.convert("RGBA")
        return image


class ImageCropAndResize(DataProcessingOperator):
    def __init__(self, height=None, width=None, max_pixels=None, height_division_factor=1, width_division_factor=1):
        self.height = height
        self.width = width
        self.max_pixels = max_pixels
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor

    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image
    
    def get_height_width(self, image):
        if self.height is None or self.width is None:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width
    
    def __call__(self, data: Image.Image):
        image = self.crop_and_resize(data, *self.get_height_width(data))
        return image


class ToList(DataProcessingOperator):
    def __call__(self, data):
        return [data]
    

class FrameSamplerByRateMixin:
    def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1, frame_rate=24, fix_frame_rate=False):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.frame_rate = frame_rate
        self.fix_frame_rate = fix_frame_rate

    def get_reader(self, data: str):
        return imageio.get_reader(data)

    def get_available_num_frames(self, reader):
        # NOTE on cost: `reader.count_frames()` shells out to `ffmpeg -c copy -f null -`,
        # i.e. a full demux pass over the whole file. Measured 0.7-2.5s on a 300-frame 540p
        # clip -- comparable to decoding it. Every call site below therefore avoids it
        # whenever the answer is obtainable from the container metadata instead.
        if not self.fix_frame_rate:
            return reader.count_frames()
        meta_data = reader.get_meta_data()
        if "duration" in meta_data:
            duration = meta_data["duration"]
        else:
            # Only this fallback genuinely needs the frame count. imageio's ffmpeg reader
            # always reports `duration`, so in practice the demux pass is never paid here.
            duration = int(reader.count_frames()) / meta_data["fps"]
        return int(math.floor(duration * self.frame_rate))

    def get_num_frames(self, reader):
        total_frames = int(self.get_available_num_frames(reader))
        if self.num_frames is None:
            # Load the full clip; snap down to the temporal division factor (4n+1 for Wan).
            num_frames = total_frames
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
            return num_frames
        num_frames = self.num_frames
        if total_frames < num_frames:
            num_frames = total_frames
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames

    def map_single_frame_id(self, new_sequence_id: int, raw_frame_rate: float, total_raw_frames: int = None) -> int:
        # `total_raw_frames` is now optional: obtaining it costs a full demux pass (see
        # get_available_num_frames), and its only use is clamping the last frame. Callers
        # that would rather guard the read itself (LoadVideo catches the IndexError, which
        # also covers a container whose `duration` OVER-states the real length) omit it.
        if not self.fix_frame_rate:
            return new_sequence_id
        target_time_in_seconds = new_sequence_id / self.frame_rate
        raw_frame_index_float = target_time_in_seconds * raw_frame_rate
        frame_id = int(round(raw_frame_index_float))
        if total_raw_frames is not None:
            frame_id = min(frame_id, total_raw_frames - 1)
        return frame_id


class LoadVideo(DataProcessingOperator, FrameSamplerByRateMixin):
    def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1, frame_processor=lambda x: x, frame_rate=24, fix_frame_rate=False):
        FrameSamplerByRateMixin.__init__(self, num_frames, time_division_factor, time_division_remainder, frame_rate, fix_frame_rate)
        # frame_processor is build in the video loader for high efficiency.
        self.frame_processor = frame_processor

    def __call__(self, data: str):
        reader = self.get_reader(data)
        raw_frame_rate = reader.get_meta_data()['fps']
        if self.fix_frame_rate and raw_frame_rate < self.frame_rate - 1e-6:
            # Resampling UP duplicates frames: consecutive outputs become identical, so the
            # inter-frame motion is zero across those pairs. Anything that reads velocity off
            # adjacent frames (e.g. E2E-TTT's multi-frame anchor block) is silently poisoned.
            # Resampling cannot manufacture the missing frames -- such clips should be
            # filtered out of the dataset instead.
            warnings.warn(
                f"{data}: source {raw_frame_rate}fps is below the target {self.frame_rate}fps; "
                f"fix_frame_rate will DUPLICATE frames, destroying inter-frame motion. "
                f"Filter this clip out rather than resampling it up."
            )
        num_frames = self.get_num_frames(reader)
        frames, truncated = [], False
        for frame_id in range(num_frames):
            frame_id = self.map_single_frame_id(frame_id, raw_frame_rate)
            try:
                frame = reader.get_data(frame_id)
            except IndexError:
                # `num_frames` can over-shoot the real end when it came from the container's
                # `duration` rather than a demux pass. Stop at the true last frame instead of
                # crashing; the snap below restores the temporal division invariant.
                truncated = True
                break
            frame = Image.fromarray(frame)
            frame = self.frame_processor(frame)
            frames.append(frame)
        reader.close()
        if truncated:
            # Only on the truncated path: get_num_frames returns `self.num_frames` verbatim
            # when the clip is long enough, and that value is not required to satisfy the
            # division factor (e.g. --num_frames 196). Snapping unconditionally would silently
            # change such configs.
            while len(frames) > 1 and len(frames) % self.time_division_factor != self.time_division_remainder:
                frames.pop()
        return frames


class SequencialProcess(DataProcessingOperator):
    def __init__(self, operator=lambda x: x):
        self.operator = operator
        
    def __call__(self, data):
        return [self.operator(i) for i in data]


class LoadGIF(DataProcessingOperator):
    def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1, frame_processor=lambda x: x):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        # frame_processor is build in the video loader for high efficiency.
        self.frame_processor = frame_processor

    def get_num_frames(self, path):
        num_frames = self.num_frames
        images = iio.imread(path, mode="RGB")
        if num_frames is None or len(images) < num_frames:
            num_frames = len(images)
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames
        
    def __call__(self, data: str):
        num_frames = self.get_num_frames(data)
        frames = []
        images = iio.imread(data, mode="RGB")
        for img in images:
            frame = Image.fromarray(img)
            frame = self.frame_processor(frame)
            frames.append(frame)
            if len(frames) >= num_frames:
                break
        return frames


class RouteByExtensionName(DataProcessingOperator):
    def __init__(self, operator_map):
        self.operator_map = operator_map
        
    def __call__(self, data: str):
        file_ext_name = data.split(".")[-1].lower()
        for ext_names, operator in self.operator_map:
            if ext_names is None or file_ext_name in ext_names:
                return operator(data)
        raise ValueError(f"Unsupported file: {data}")


class RouteByType(DataProcessingOperator):
    def __init__(self, operator_map):
        self.operator_map = operator_map
        
    def __call__(self, data):
        for dtype, operator in self.operator_map:
            if dtype is None or isinstance(data, dtype):
                return operator(data)
        raise ValueError(f"Unsupported data: {data}")


class LoadTorchPickle(DataProcessingOperator):
    def __init__(self, map_location="cpu"):
        self.map_location = map_location
        
    def __call__(self, data):
        return torch.load(data, map_location=self.map_location, weights_only=False)


class ToAbsolutePath(DataProcessingOperator):
    def __init__(self, base_path=""):
        self.base_path = base_path
        
    def __call__(self, data):
        return os.path.join(self.base_path, data)


class LoadAudio(DataProcessingOperator):
    def __init__(self, sr=16000):
        self.sr = sr
        import librosa
        self.audio_loader = librosa.load
    
    def __call__(self, data: str):
        input_audio, sample_rate = self.audio_loader(data, sr=self.sr)
        return input_audio


class LoadAudioWithTorchaudio(DataProcessingOperator, FrameSamplerByRateMixin):

    def __init__(self, num_frames=121, time_division_factor=8, time_division_remainder=1, frame_rate=24, fix_frame_rate=True):
        FrameSamplerByRateMixin.__init__(self, num_frames, time_division_factor, time_division_remainder, frame_rate, fix_frame_rate)
        import torchaudio
        self.audio_loader = torchaudio.load

    def __call__(self, data: str):
        try:
            reader = self.get_reader(data)
            num_frames = self.get_num_frames(reader)
            duration = num_frames / self.frame_rate
            waveform, sample_rate = self.audio_loader(data)
            target_samples = int(duration * sample_rate)
            current_samples = waveform.shape[-1]
            if current_samples > target_samples:
                waveform = waveform[..., :target_samples]
            elif current_samples < target_samples:
                padding = target_samples - current_samples
                waveform = torch.nn.functional.pad(waveform, (0, padding))
            return waveform, sample_rate
        except:
            warnings.warn(f"Cannot load audio in {data}. The audio will be `None`.")
            return None


class LoadPureAudioWithTorchaudio(DataProcessingOperator):

    def __init__(self, target_sample_rate=None, target_duration=None):
        self.target_sample_rate = target_sample_rate
        self.target_duration = target_duration
        self.resample = True if target_sample_rate is not None else False
        from diffsynth.utils.data.audio import read_audio
        self.audio_loader = read_audio

    def __call__(self, data: str):
        try:
            waveform, sample_rate = self.audio_loader(data, resample=self.resample, resample_rate=self.target_sample_rate)
            if self.target_duration is not None:
                target_samples = int(self.target_duration * sample_rate)
                current_samples = waveform.shape[-1]
                if current_samples > target_samples:
                    waveform = waveform[..., :target_samples]
                elif current_samples < target_samples:
                    padding = target_samples - current_samples
                    waveform = torch.nn.functional.pad(waveform, (0, padding))
            return waveform, sample_rate
        except Exception as e:
            print(f"Cannot load audio in {data} due to {e}. The audio will be `None`.")
            return None
