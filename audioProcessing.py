import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.io import wavfile
import soundfile as sf

# 1. Cargar el audio en formato .wav
audio_file = 'diegoTijo.wav'

# Intentar cargar con librosa / scipy
try:
    y, sr = librosa.load(audio_file, sr=None)
except Exception:
    sr, y = wavfile.read(audio_file)
    if y.ndim > 1:
        y = y.mean(axis=1) # Convertir estéreo a mono
    y = y.astype(np.float32)
    y /= np.max(np.abs(y)) # Normalizar

# 2. Filtrar intervalos de silencio
non_silent_indices = librosa.effects.split(y, top_db=20)
if len(non_silent_indices) > 0:
    filtered_audio = np.concatenate([y[start:end] for start, end in non_silent_indices])
else:
    filtered_audio = y

# 3. Conversión Analógico-Digital (ADC)
desired_sample_rate = 8000
bit_depth = 8

resampled_audio = librosa.resample(filtered_audio, orig_sr=sr, target_sr=desired_sample_rate)
normalized_audio = resampled_audio / np.max(np.abs(resampled_audio))

max_amplitude = 2 ** (bit_depth - 1) - 1
quantized_audio = np.round(normalized_audio * max_amplitude).astype(np.int16)

# Guardar la señal cuantizada en el rango completo para que se escuche fuerte
sf.write('voz_codificada.wav', (quantized_audio * 256).astype(np.int16), desired_sample_rate)

# 4. Transformada Rápida de Fourier
n = len(resampled_audio)
T = 1.0 / desired_sample_rate
yf = fft(resampled_audio)
xf = fftfreq(n, T)[:n//2]

amplitude = 2.0 / n * np.abs(yf[:n//2])

# 5. Identificar Frecuencia Dominante y Rango
dominant_frequency_index = np.argmax(amplitude)
dominant_frequency = xf[dominant_frequency_index]

threshold = 0.1 * np.max(amplitude)
significant_indices = np.where(amplitude > threshold)[0]
freq_min = xf[significant_indices[0]]
freq_max = xf[significant_indices[-1]]

print(f"Frecuencia Dominante: {dominant_frequency:.2f} Hz")
print(f"Rango de Frecuencias: {freq_min:.2f} Hz - {freq_max:.2f} Hz")

# 6. Visualización
plt.figure(figsize=(10, 6))
plt.plot(xf, amplitude, color='g', label='Espectro de Frecuencia')
plt.axvline(x=dominant_frequency, color='r', linestyle='--', label=f'Frecuencia Dominante: {dominant_frequency:.2f} Hz')
plt.axvspan(freq_min, freq_max, color='y', alpha=0.3, label=f'Rango: {freq_min:.2f} - {freq_max:.2f} Hz')
plt.title('Espectro de Frecuencia de la Voz')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Amplitud')
plt.xlim(0, 1500)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()