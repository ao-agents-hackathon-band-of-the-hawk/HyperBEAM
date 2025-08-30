from generator import Segment, load_csm_1b
import torchaudio
import torch

seed = 42

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

generator = load_csm_1b(device=device)

speakers = [0]
transcripts = [
    "In a 1997 AI class at UT Austin, a neural net playing infinite board tic-tac-toe found an unbeatable strategy. Choose moves billions of squares away, causing your opponents to run out of memory and crash.",
]
audio_paths = [
    "utterance_0.mp3",
]

def load_audio(audio_path):
    audio_tensor, sample_rate = torchaudio.load(audio_path)
    audio_tensor = audio_tensor.mean(dim=0)  # Average channels to mono
    audio_tensor = torchaudio.functional.resample(
        audio_tensor.squeeze(0), orig_freq=sample_rate, new_freq=generator.sample_rate
    )
    return audio_tensor

segments = [
    Segment(text=transcript, speaker=speaker, audio=load_audio(audio_path))
    for transcript, speaker, audio_path in zip(transcripts, speakers, audio_paths)
]
audio = generator.generate(
    text="Hello Am I audible, I wanted you guys to tell me if my voice changes alot, okay thanks!",
    speaker=0,
    context=segments,
    max_audio_length_ms=10_000,
)
print("haii")
torchaudio.save("audio.mp3", audio.unsqueeze(0).cpu(), generator.sample_rate)
print("Audio saved to audio.wav")
