# Chapter 8: Speech-to-text

**Goal:** let customers speak instead of type, by putting a transcription step in front of the pipeline.

## 8.1 Whisper

**Whisper** is OpenAI's speech recognition model. It handles accents, background noise and ~95 languages
(including Hindi, Telugu and Tamil, and mixtures with English). Two ways to run it:

| `STT_BACKEND` | How | Pros | Cons |
|---|---|---|---|
| `local` | `openai-whisper` package on your computer | Free, private | Needs FFmpeg; first run downloads the model |
| `openai` | OpenAI transcription API | No FFmpeg, fast | Small cost, audio leaves your computer |

Local model sizes (`WHISPER_MODEL`): `tiny` (fastest) → `base` (default) → `small` → `medium` → `large`
(most accurate). For Indian languages, `small` or larger works noticeably better.

## 8.2 The code

Create `step5_speech_to_text.py`:

````python
"""
STEP 5: Speech-to-text with Whisper.

Customers often prefer to speak rather than type. Whisper (by OpenAI) converts
audio into text. Two ways to run it, chosen by STT_BACKEND in .env:

  local  -> the open-source `openai-whisper` package runs on your own computer.
            Free and private, but needs FFmpeg installed. First run downloads the model.
  openai -> sends the audio to OpenAI's transcription API. No FFmpeg needed, small cost.

Whisper also handles Hindi, Telugu, Tamil and ~95 other languages, and can mix
them with English.

The transcription step is wrapped in a RunnableLambda so it plugs straight into
the Step 4 pipeline with the | operator:   transcribe | support_pipeline
"""
import os
import sys
from functools import lru_cache
from pathlib import Path

from langchain_core.runnables import RunnableLambda

from step4_dag_workflow import support_pipeline


@lru_cache(maxsize=1)
def _load_local_whisper():
    import whisper  # imported lazily: it's a big package

    model_name = os.getenv("WHISPER_MODEL", "base")  # tiny, base, small, medium, large
    print(f"Loading Whisper '{model_name}' model (first time downloads it)...")
    return whisper.load_model(model_name)


def transcribe(audio_path: str, backend: str | None = None) -> str:
    """Convert an audio file (wav, mp3, m4a, webm...) into text."""
    if not audio_path or not Path(audio_path).exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    backend = (backend or os.getenv("STT_BACKEND", "local")).lower()

    if backend == "openai":
        from openai import OpenAI

        with open(audio_path, "rb") as audio_file:
            result = OpenAI().audio.transcriptions.create(
                model=os.getenv("OPENAI_STT_MODEL", "whisper-1"), file=audio_file
            )
        return result.text.strip()

    # fp16=False avoids a warning on computers without a GPU
    result = _load_local_whisper().transcribe(audio_path, fp16=False)
    return result["text"].strip()


# Speech -> text -> full support pipeline, as one chain
voice_pipeline = (
    RunnableLambda(lambda x: {"query": transcribe(x["audio_path"]), "history": x.get("history") or "(none)"})
    | RunnableLambda(lambda x: {**support_pipeline.invoke(x), "transcript": x["query"]})
)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python step5_speech_to_text.py path/to/recording.wav")
        print("Tip: record a voice note on your phone, e.g. 'Where is my order O R D 1002?'")
        sys.exit(1)

    result = voice_pipeline.invoke({"audio_path": sys.argv[1]})
    print(f"Transcript : {result['transcript']}")
    print(f"Route      : {result['route']}")
    print(f"Reply      : {result['answer']}")
````

The key idea is the last part: wrapping `transcribe` in a `RunnableLambda` lets a non-LLM step join the chain
with `|`. Anything (speech recognition, a database query, an API call) can become a pipeline step this way.

## 8.3 Run it

Record a short voice note on your phone (for example *"Where is my order O R D one zero zero two?"*), copy
it into the project folder, then:

```powershell
python step5_speech_to_text.py my_question.m4a
```

## Checkpoint

```
Transcript : Where is my order ORD1002?
Route      : quick_lookup
Reply      : ...
```

If you see an FFmpeg error, close the terminal, open a new one, activate the venv and try again (a new
terminal picks up the updated PATH). Or set `STT_BACKEND=openai`. Spoken IDs sometimes come out as
"1002" without the letters; `tools._clean_id` turns that into `ORD1002`.

**Next: [Chapter 9: Web app →](09-web-app.md)**
