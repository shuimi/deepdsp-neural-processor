import simpleaudio as sa
from neural_model.dsp_core import *

SECONDS = 2
time = np.arange(0, SAMPLE_RATE * SECONDS, 1)


audio = detune(time, sawtooth_wave, freq=55, voices_amount=2, detune_st=0.1, blend=1)
# audio = lowpass(audio, 0.4)


play = sa.play_buffer(
    to16bit(audio),
    num_channels=1,
    bytes_per_sample=2,
    sample_rate=SAMPLE_RATE
)
play.wait_done()
