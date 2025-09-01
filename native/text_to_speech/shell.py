import argparse
from generator import load_csm_1b
import torchaudio
import torch

def generate_audio(text, output, speaker, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    generator = load_csm_1b(device=device)
    
    audio = generator.generate(
        text=text,
        speaker=speaker,
        context=[],
        max_audio_length_ms=10000,
    )
    
    torchaudio.save(output, audio.unsqueeze(0).cpu(), generator.sample_rate)
    print(f"Audio saved to {output}")

def main():
    parser = argparse.ArgumentParser(description='Text to Speech CLI')
    parser.add_argument('--text', required=True, help='Text to convert to speech')
    parser.add_argument('--output', required=True, help='Output audio file path')
    parser.add_argument('--speaker', type=int, default=0, help='Speaker ID (default: 0)')
    parser.add_argument('--device', default=None, help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    generate_audio(args.text, args.output, args.speaker, args.device)

if __name__ == "__main__":
    main()