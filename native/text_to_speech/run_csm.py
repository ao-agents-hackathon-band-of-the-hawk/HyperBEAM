import os
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from generator import load_csm_1b, Segment
from dataclasses import dataclass
import json

seed = 42

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Disable Triton compilation
os.environ["NO_TORCH_COMPILE"] = "1"

# --- Session Setup ---
SESSION_ID = "test-session"
SESSIONS_DIR = "../../sessions"
USER_AUDIOS_PATH = os.path.join(SESSIONS_DIR, SESSION_ID, "user-audios")
RESPONSE_AUDIOS_PATH = os.path.join(SESSIONS_DIR, SESSION_ID, "response-audios")

# Create directories for the session
os.makedirs(USER_AUDIOS_PATH, exist_ok=True)
os.makedirs(RESPONSE_AUDIOS_PATH, exist_ok=True)
print(f"Session directories created at: {os.path.join(SESSIONS_DIR, SESSION_ID)}")

# Default prompts are available at https://hf.co/sesame/csm-1b
prompt_filepath_conversational_a = hf_hub_download(
    repo_id="sesame/csm-1b",
    filename="prompts/conversational_a.wav"
)
prompt_filepath_conversational_b = hf_hub_download(
    repo_id="sesame/csm-1b",
    filename="prompts/conversational_b.wav"
)

SPEAKER_PROMPTS = {
    "conversational_a": {
        "text": (
            "like revising for an exam I'd have to try and like keep up the momentum because I'd "
            "start really early I'd be like okay I'm gonna start revising now and then like "
            "you're revising for ages and then I just like start losing steam I didn't do that "
            "for the exam we had recently to be fair that was a more of a last minute scenario "
            "but like yeah I'm trying to like yeah I noticed this yesterday that like Mondays I "
            "sort of start the day with this not like a panic but like a"
        ),
        "audio": prompt_filepath_conversational_a
    },
    "conversational_b": {
        "text": (
            "like a super Mario level. Like it's very like high detail. And like, once you get "
            "into the park, it just like, everything looks like a computer game and they have all "
            "these, like, you know, if, if there's like a, you know, like in a Mario game, they "
            "will have like a question block. And if you like, you know, punch it, a coin will "
            "come out. So like everyone, when they come into the park, they get like this little "
            "bracelet and then you can go punching question blocks around."
        ),
        "audio": prompt_filepath_conversational_b
    }
}

def load_prompt_audio(audio_path: str, target_sample_rate: int) -> torch.Tensor:
    audio_tensor, sample_rate = torchaudio.load(audio_path)
    audio_tensor = audio_tensor.squeeze(0)
    # Resample is lazy so we can always call it
    audio_tensor = torchaudio.functional.resample(
        audio_tensor, orig_freq=sample_rate, new_freq=target_sample_rate
    )
    return audio_tensor

def prepare_prompt(text: str, speaker: int, audio_path: str, sample_rate: int) -> Segment:
    audio_tensor = load_prompt_audio(audio_path, sample_rate)
    return Segment(text=text, speaker=speaker, audio=audio_tensor)

def save_utterance(text: str, audio: torch.Tensor, speaker_id: int, counter: int, sample_rate: int):
    """Saves the audio and updates the transcript list."""
    if speaker_id == 0:
        path = USER_AUDIOS_PATH
    else:
        path = RESPONSE_AUDIOS_PATH
    
    filename = os.path.join(path, f"{counter}.wav")
    torchaudio.save(filename, audio.unsqueeze(0).cpu(), sample_rate)
    return text

def main():
    # Select the best available device, skipping MPS due to float64 limitations
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # Load model
    generator = load_csm_1b(device)

    # --- Generate and Save Conversation History ---
    print(f"\nGenerating conversation history for session: {SESSION_ID}")

    # Prepare and save initial prompts
    prompt_a = prepare_prompt(
        SPEAKER_PROMPTS["conversational_a"]["text"],
        0, # Speaker 0 (User)
        SPEAKER_PROMPTS["conversational_a"]["audio"],
        generator.sample_rate
    )
    user_transcripts = [save_utterance(prompt_a.text, prompt_a.audio, 0, 0, generator.sample_rate)]
    
    prompt_b = prepare_prompt(
        SPEAKER_PROMPTS["conversational_b"]["text"],
        1, # Speaker 1 (Response)
        SPEAKER_PROMPTS["conversational_b"]["audio"],
        generator.sample_rate
    )
    response_transcripts = [save_utterance(prompt_b.text, prompt_b.audio, 1, 0, generator.sample_rate)]

    prompt_segments = [prompt_a, prompt_b]
    generated_segments = []
    
    user_file_counter = 1
    response_file_counter = 1

    conversation = [
        {"text": "Hey how are you doing?", "speaker_id": 0},
        {"text": "Pretty good, pretty good. How about you?", "speaker_id": 1},
        {"text": "I'm great! So happy to be speaking with you today.", "speaker_id": 0},
        {"text": "Me too! This is some cool stuff, isn't it?", "speaker_id": 1}
    ]

    for utterance in conversation:
        current_context = prompt_segments + generated_segments
        print(f"Generating (Speaker {utterance['speaker_id']}): \"{utterance['text']}\"")
        
        audio_tensor = generator.generate(
            text=utterance['text'],
            speaker=utterance['speaker_id'],
            context=current_context,
            max_audio_length_ms=10_000,
        )
        
        # Save the generated audio and transcript
        if utterance['speaker_id'] == 0:
            user_transcripts.append(save_utterance(utterance['text'], audio_tensor, 0, user_file_counter, generator.sample_rate))
            user_file_counter += 1
        else:
            response_transcripts.append(save_utterance(utterance['text'], audio_tensor, 1, response_file_counter, generator.sample_rate))
            response_file_counter += 1
        
        # Add to context for the next turn
        generated_segments.append(Segment(text=utterance['text'], speaker=utterance['speaker_id'], audio=audio_tensor))

    # Write the final transcript lists to JSON files
    with open(os.path.join(USER_AUDIOS_PATH, "string-list.json"), "w") as f:
        json.dump(user_transcripts, f, indent=4)
    with open(os.path.join(RESPONSE_AUDIOS_PATH, "string-list.json"), "w") as f:
        json.dump(response_transcripts, f, indent=4)
        
    print("\nConversation history saved successfully.")
    print(f"User files in: {USER_AUDIOS_PATH}")
    print(f"Response files in: {RESPONSE_AUDIOS_PATH}")

    # --- Generate a final utterance using the full saved context ---
    print("\nGenerating a final audio file using the full session context...")
    
    final_utterance_text = "That's fascinating. Tell me more about the Mario game."
    final_speaker_id = 0 # User asking a follow-up
    
    final_audio = generator.generate(
        text=final_utterance_text,
        speaker=final_speaker_id,
        context=prompt_segments + generated_segments, # Use the full history
        max_audio_length_ms=10_000,
    )
    
    final_output_path = "final_contextual_generation.wav"
    torchaudio.save(
        final_output_path,
        final_audio.unsqueeze(0).cpu(),
        generator.sample_rate
    )
    print(f"Successfully generated final audio: {final_output_path}")
    print("This file's voice should be consistent with Speaker 0 from the conversation.")

if __name__ == "__main__":
    main()