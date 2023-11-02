import sounddevice as sd
import numpy as np
from deepspeech import Model
from translate import Translator

def record_audio(seconds=10, samplerate=16000):
    print("Recording...")
    audio = sd.rec(int(seconds * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()
    print("Recording finished")
    return np.array(audio).flatten()

def speech_to_text(audio, model_file, scorer_file):
    model = Model(model_file)
    model.enableExternalScorer(scorer_file)

    text = model.stt(audio)
    return text

def translate_text(text, target_language="zh"):
    translator = Translator(to_lang=target_language)
    translation = translator.translate(text)
    return translation

if __name__ == "__main__":
    model_file = "deepspeech-0.9.3-models.pbmm"
    scorer_file = "deepspeech-0.9.3-models.scorer"

    audio = record_audio()
    text = speech_to_text(audio, model_file, scorer_file)
    print("Text: {}".format(text))

    translation = translate_text(text)
    print("Translation: {}".format(translation))
