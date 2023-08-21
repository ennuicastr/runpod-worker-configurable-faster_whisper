<div align="center">

<h1>Faster Whisper | Worker</h1>

This is an expanded version of RunPod's faster-whisper endpoint, with added
features. The added features are:
 * Support for handling multiple audio files in one request, which can use GPU
   time more efficiently.
 * Support for disabling the VTT/SRT transcription and including only the JSON
   output.
 * Access to faster-whisper's `word_timestamps` option, which gives word-by-word
   transcriptions.
 * Access to faster-whisper's `vad_filter` option, which is much, MUCH faster at
   skipping audio with no words than Whisper itself is.
 * The ability to remove extraneous details such as confidence.

The original, RunPod description of this repository: This repository contains the [Faster Whisper](https://github.com/guillaumekln/faster-whisper) Worker for RunPod. The Whisper Worker is designed to process audio files using various Whisper models, with options for transcription formatting, language translation, and more. It's part of the RunPod Workers collection aimed at providing diverse functionality for endpoint processing.

[Endpoint Docs](https://docs.runpod.io/reference/faster-whisper)

[Docker Image](https://hub.docker.com/r/runpod/ai-api-faster-whisper)

</div>

## Model Inputs

| Input                               | Type  | Description                                                                                                 |
|-------------------------------------|-------|-------------------------------------------------------------------------------------------------------------|
| `audio`                             | Path  | Audio file                                                                                                  |
| `audios`                            | array of paths | Multiple audio files. One of `audio` or `audios` must be set, not both.                            |
| `model`                             | str   | Choose a Whisper model. Choices: "tiny", "base", "small", "medium", "large-v1", "large-v2". Default: "base" |
| `transcription`                     | str   | Choose the format for the transcription. Choices: "none", "srt", "vtt". Default: "none". Note that a JSON version of the transcription is always included. |
| `translate`                         | bool  | Translate the text to English when set to True. Default: False                                              |
| `language`                          | str   | Language spoken in the audio, specify None to perform language detection. Default: None                     |
| `temperature`                       | float | Temperature to use for sampling. Default: 0                                                                 |
| `best_of`                           | int   | Number of candidates when sampling with non-zero temperature. Default: 5                                    |
| `beam_size`                         | int   | Number of beams in beam search, only applicable when temperature is zero. Default: 5                        |
| `patience`                          | float | Optional patience value to use in beam decoding. Default: None                                              |
| `length_penalty`                    | float | Optional token length penalty coefficient (alpha). Default: None                                            |
| `suppress_tokens`                   | str   | Comma-separated list of token ids to suppress during sampling. Default: "-1"                                |
| `initial_prompt`                    | str   | Optional text to provide as a prompt for the first window. Default: None                                    |
| `condition_on_previous_text`        | bool  | If True, provide the previous output of the model as a prompt for the next window. Default: True            |
| `temperature_increment_on_fallback` | float | Temperature to increase when falling back when the decoding fails. Default: 0.2                             |
| `compression_ratio_threshold`       | float | If the gzip compression ratio is higher than this value, treat the decoding as failed. Default: 2.4         |
| `logprob_threshold`                 | float | If the average log probability is lower than this value, treat the decoding as failed. Default: -1.0        |
| `no_speech_threshold`               | float | If the probability of the token is higher than this value, consider the segment as silence. Default: 0.6    |
| `word_timestamps`                   | bool  | Include per-word timestamps as `result.words`. Default: false                                               |
| `vad_filter`                        | bool  | Use the VAD filter to avoid transcribing non-speech. Default: true                                          |
| `detailed`                          | bool  | Include irrelevant details in the segments returned. Default: false                                         |

## Test Inputs

The following inputs can be used for testing the model:

```json
{
    "input": {
        "audio": "https://github.com/runpod-workers/sample-inputs/raw/main/audio/gettysburg.wav"
    }
}
```
