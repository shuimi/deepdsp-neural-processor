import simpleaudio as sa
from neural_model.dsp_core import *

SECONDS = 2
time = np.arange(0, SAMPLE_RATE * SECONDS, 1)


audio = multiwave_oscillator(time, freq=440)


play = sa.play_buffer(
    to16bit(audio),
    num_channels=1,
    bytes_per_sample=2,
    sample_rate=SAMPLE_RATE
)
play.wait_done()
