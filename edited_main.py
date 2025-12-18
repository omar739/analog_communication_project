import streamlit as st
import matplotlib.pyplot as plt
import soundfile as sf
import librosa.display
import base64
import numpy as np
import io
import soundfile as sf
from DSB import modulation_dsbsc , demodulation_dsbsc , modulation_dsbtc , demodulation_dsbtc

@st.cache_data
def load_background(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


@st.cache_data
def load_audio(file):
    y, sr = librosa.load(file, sr=None)
    return y, sr*5


@st.cache_data
def compute_fft(y, sr):
    N = len(y)
    fft = np.fft.fft(y)
    freq = np.fft.fftfreq(N, 1 / sr)
    magnitude = np.abs(fft) / N
    pos_idx = np.where(freq >= 0)
    return freq[pos_idx], magnitude[pos_idx]


st.set_page_config(layout="centered", page_icon="📡", page_title="Analog Communication")

b64 = load_background("76826.jpg")

st.markdown(f"""
<style>
.stApp {{
    background-image: url("data:image/jpg;base64,{b64}");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
}}
</style>
""", unsafe_allow_html=True)


st.title("Analog Communication Project")

st.image("Screenshot from 2025-12-12 21-58-21.png")

with open("text.txt", "+r") as file:
    txt = file.read()
st.subheader("What is Analog communication ? ")
st.write(txt)




st.markdown("""
<h2 style='background-color: #1f4433; color: white; padding: 5px; border-radius:5px;'>
    Team Names
</h2>""", unsafe_allow_html=True)

st.subheader("`Omar Ahmed Abbas - 22010944`")
st.subheader("`Ebrahim Alaa Mohammed - 22010475`")
st.subheader("`Salma Mohammed Ali - 22010812`")
st.subheader("`Mona Yasser Awad - 22011272`")

st.markdown("""
<h2 style='background-color: #1f4433; color: white; padding: 5px; border-radius:5px;'>
    Upload your audio
</h2>""", unsafe_allow_html=True)

audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg"])
if audio_file is not None:
    y, sr = load_audio(audio_file)

    fig1, ax1 = plt.subplots(figsize=(12, 4))
    ax1.tick_params(colors="white")
    ax1.set_facecolor("#222222")
    fig1.patch.set_facecolor("#222222")
    for spine in ax1.spines.values():
        spine.set_color("white")

    librosa.display.waveshow(y, sr=sr, ax=ax1, color="#00ff0d")
    ax1.set_title("Waveform", color="white")
    st.pyplot(fig1)

    freq, magnitude = compute_fft(y, sr)

    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ax2.tick_params(colors="white")
    ax2.set_facecolor("#222222")
    fig2.patch.set_facecolor("#222222")
    for spine in ax2.spines.values():
        spine.set_color("white")

    ax2.plot(freq, magnitude, color="#00ffff")
    ax2.set_title("Frequency Spectrum (FFT)", color="white")
    ax2.set_xlabel("Frequency (Hz)", color="white")
    ax2.set_ylabel("Magnitude", color="white")
    st.pyplot(fig2)

    if st.button("Play Audio"):
        st.audio(audio_file)

st.markdown("""
<h2 style='background-color: #1f4433; color: white; padding: 5px; border-radius:5px;'>
    Choose your modulation Type
</h2>""", unsafe_allow_html=True)

options = [
    "Double side band Suppressed Carrier DSB-SC",
    "Double Side Band Transmitted Carrier DSB-TC",
    "Single Side Band Suppressed Carrier SSB-SC",
    "Single Side Band Transmitted Carrier SSB-TC",
    "Frequency modulation FM"
]
choise = st.selectbox("Choose modulation type", options)

if choise == "Double side band Suppressed Carrier DSB-SC" :
    with open("DSB_SC.txt","r+") as file:
        text = file.read()
    st.code(text)
elif choise == "Double Side Band Transmitted Carrier DSB-TC" :
    with open("DSB_TC.txt","r+") as file:
        text = file.read()
    st.code(text)


st.header("choose your parameters")
if choise == "Double side band Suppressed Carrier DSB-SC":
    max_fc = sr // 2
    st.header("Transmitter of DSB-SC")
    fc = st.number_input(
        label=f"Enter Carrier frequency (max {max_fc} Hz)", 
        min_value=1, max_value=max_fc, value=1000
    )
    amp = st.number_input(label="Enter Amplitude of Carrier", min_value=1, max_value=10, value=1)
    phase = st.number_input(label="Enter Phase of carrier", min_value=0.0, max_value=2*np.pi, value=0.0, step=0.01)
    t = np.arange(0, len(y)) / sr
    carrier = amp * np.cos(2 * np.pi * fc * t + phase)
    
    button = st.button("Show Carrier")
    if button:

        fig3, ax3 = plt.subplots(figsize=(12, 4))
        ax3.tick_params(colors="white")
        ax3.set_facecolor("#222222")
        fig3.patch.set_facecolor("#222222")
        for spine in ax3.spines.values():
            spine.set_color("white")
        ax3.plot(t, carrier, color="#ff9900")
        ax3.set_title("Carrier Signal", color="white")
        ax3.set_xlabel("Time (s)", color="white")
        ax3.set_ylabel("Amplitude", color="white")
        st.pyplot(fig3)

        N = len(carrier)
        fft_carrier = np.fft.fft(carrier)
        freq_carrier = np.fft.fftfreq(N, 1/sr)
        magnitude_carrier = np.abs(fft_carrier) / N
        pos_idx = np.where(freq_carrier >= 0)
        freq_carrier = freq_carrier[pos_idx]
        magnitude_carrier = magnitude_carrier[pos_idx]

        fig4, ax4 = plt.subplots(figsize=(12, 4))
        ax4.tick_params(colors="white")
        ax4.set_facecolor("#222222")
        fig4.patch.set_facecolor("#222222")
        for spine in ax4.spines.values():
            spine.set_color("white")
        ax4.plot(freq_carrier, magnitude_carrier, color="#ff00ff")
        ax4.set_title("Frequency Spectrum of Carrier", color="white")
        ax4.set_xlabel("Frequency (Hz)", color="white")
        ax4.set_ylabel("Magnitude", color="white")
        ax4.set_xlim(0, max_fc*1.2)
        st.pyplot(fig4)
        
    mod_signal, freq_mod, mag_mod = modulation_dsbsc(y, carrier, sr)    
    button_mod = st.button("Compute DSB-SC Modulated Signal")
    if button_mod:
                
        fig_mod, ax_mod = plt.subplots(figsize=(12, 4))
        ax_mod.tick_params(colors="white")
        ax_mod.set_facecolor("#222222")
        fig_mod.patch.set_facecolor("#222222")
        for spine in ax_mod.spines.values():
            spine.set_color("white")
        ax_mod.plot(np.arange(0, len(mod_signal))/sr, mod_signal, color="#00ff99")
        ax_mod.set_title("DSB-SC Modulated Signal (Time Domain)", color="white")
        ax_mod.set_xlabel("Time (s)", color="white")
        ax_mod.set_ylabel("Amplitude", color="white")
        st.pyplot(fig_mod)
                
        fig_mod_fft, ax_mod_fft = plt.subplots(figsize=(12, 4))
        ax_mod_fft.tick_params(colors="white")
        ax_mod_fft.set_facecolor("#222222")
        fig_mod_fft.patch.set_facecolor("#222222")
        for spine in ax_mod_fft.spines.values():
            spine.set_color("white")
        ax_mod_fft.plot(freq_mod, mag_mod, color="#ff3399")
        ax_mod_fft.set_title("DSB-SC Modulated Signal (Frequency Domain)", color="white")
        ax_mod_fft.set_xlabel("Frequency (Hz)", color="white")
        ax_mod_fft.set_ylabel("Magnitude", color="white")
        st.pyplot(fig_mod_fft)
    
    st.header("Receiver of DSB-SC")

    fcD = st.number_input(
        label=f"Enter Carrier frequency (max {max_fc} Hz)", 
        min_value=1, max_value=max_fc, value=1000, key="fcD"
    )
    ampD = st.number_input(label="Enter Amplitude of Carrier", min_value=1, max_value=10, value=1, key="ampD")
    phaseD = st.number_input(label="Enter Phase of carrier", min_value=0.0, max_value=2*np.pi, value=0.0, step=0.01, key="phaseD")
    tD = np.arange(0, len(y)) / sr
    carrierD = ampD * np.cos(2 * np.pi * fcD * tD + phaseD)

    BWD = st.number_input("Enter BW of LPF", min_value=1, max_value=max_fc, value=1000, key="BWD")
    gain = st.number_input("Enter Gain of LPF", min_value=1, max_value=10, value=1, key="gainLPF")
    SNR = st.number_input("SNR Value with dB",min_value=0,max_value=50)
    button_demod = st.button("Show Demodulated Signal")
    if button_demod:
        demod_signal, freq_demod, mag_demod = demodulation_dsbsc(mod_signal, carrierD, sr, BWD, gain, SNR)
        
        fig_demod, ax_demod = plt.subplots(figsize=(12, 4))
        ax_demod.tick_params(colors="white")
        ax_demod.set_facecolor("#222222")
        fig_demod.patch.set_facecolor("#222222")
        for spine in ax_demod.spines.values():
            spine.set_color("white")
        ax_demod.plot(np.arange(0, len(demod_signal))/sr, demod_signal, color="#00ccff")
        ax_demod.set_title("DSB-SC Demodulated Signal (Time Domain)", color="white")
        ax_demod.set_xlabel("Time (s)", color="white")
        ax_demod.set_ylabel("Amplitude", color="white")
        st.pyplot(fig_demod)
        
        fig_demod_fft, ax_demod_fft = plt.subplots(figsize=(12, 4))
        ax_demod_fft.tick_params(colors="white")
        ax_demod_fft.set_facecolor("#222222")
        fig_demod_fft.patch.set_facecolor("#222222")
        for spine in ax_demod_fft.spines.values():
            spine.set_color("white")
        ax_demod_fft.plot(freq_demod, mag_demod, color="#ff6600")
        ax_demod_fft.set_title("DSB-SC Demodulated Signal (Frequency Domain)", color="white")
        ax_demod_fft.set_xlabel("Frequency (Hz)", color="white")
        ax_demod_fft.set_ylabel("Magnitude", color="white")
        st.pyplot(fig_demod_fft)
        
        buffer = io.BytesIO()
        sf.write(buffer, demod_signal, int(sr/5), format='WAV')
        st.audio(buffer)
    


elif choise == "Double Side Band Transmitted Carrier DSB-TC":
    max_fc = sr // 2
    st.header("Transmitter of DSB-SC")
    fc = st.number_input(
        label=f"Enter Carrier frequency (max {max_fc} Hz)", 
        min_value=1, max_value=max_fc, value=1000
    )
    amp = st.number_input(label="Enter Amplitude of Carrier", min_value=1, max_value=10, value=1)
    phase = st.number_input(label="Enter Phase of carrier", min_value=0.0, max_value=2*np.pi, value=0.0, step=0.01)
    DC = st.number_input(label=f"Enter DC Shift, min is {np.max(y)}",min_value=0.0,max_value=100.0,step=1.0)
    t = np.arange(0, len(y)) / sr
    carrier = amp * np.cos(2 * np.pi * fc * t + phase)
    
    button = st.button("Show Carrier")
    if button:
        fig3, ax3 = plt.subplots(figsize=(12, 4))
        ax3.tick_params(colors="white")
        ax3.set_facecolor("#222222")
        fig3.patch.set_facecolor("#222222")
        for spine in ax3.spines.values():
            spine.set_color("white")
        ax3.plot(t, carrier, color="#ff9900")
        ax3.set_title("Carrier Signal", color="white")
        ax3.set_xlabel("Time (s)", color="white")
        ax3.set_ylabel("Amplitude", color="white")
        st.pyplot(fig3)

        N = len(carrier)
        fft_carrier = np.fft.fft(carrier)
        freq_carrier = np.fft.fftfreq(N, 1/sr)
        magnitude_carrier = np.abs(fft_carrier) / N
        pos_idx = np.where(freq_carrier >= 0)
        freq_carrier = freq_carrier[pos_idx]
        magnitude_carrier = magnitude_carrier[pos_idx]

        fig4, ax4 = plt.subplots(figsize=(12, 4))
        ax4.tick_params(colors="white")
        ax4.set_facecolor("#222222")
        fig4.patch.set_facecolor("#222222")
        for spine in ax4.spines.values():
            spine.set_color("white")
        ax4.plot(freq_carrier, magnitude_carrier, color="#ff00ff")
        ax4.set_title("Frequency Spectrum of Carrier", color="white")
        ax4.set_xlabel("Frequency (Hz)", color="white")
        ax4.set_ylabel("Magnitude", color="white")
        ax4.set_xlim(0, max_fc*1.2)
        st.pyplot(fig4)
        
    mod_signal, freq_mod, mag_mod = modulation_dsbtc(y, carrier, sr,DC)    
    button_mod = st.button("Compute DSB-SC Modulated Signal")
    if button_mod:
                
        fig_mod, ax_mod = plt.subplots(figsize=(12, 4))
        ax_mod.tick_params(colors="white")
        ax_mod.set_facecolor("#222222")
        fig_mod.patch.set_facecolor("#222222")
        for spine in ax_mod.spines.values():
            spine.set_color("white")
        ax_mod.plot(np.arange(0, len(mod_signal))/sr, mod_signal, color="#00ff99")
        ax_mod.set_title("DSB-SC Modulated Signal (Time Domain)", color="white")
        ax_mod.set_xlabel("Time (s)", color="white")
        ax_mod.set_ylabel("Amplitude", color="white")
        st.pyplot(fig_mod)
                
        fig_mod_fft, ax_mod_fft = plt.subplots(figsize=(12, 4))
        ax_mod_fft.tick_params(colors="white")
        ax_mod_fft.set_facecolor("#222222")
        fig_mod_fft.patch.set_facecolor("#222222")
        for spine in ax_mod_fft.spines.values():
            spine.set_color("white")
        ax_mod_fft.plot(freq_mod, mag_mod, color="#ff3399")
        ax_mod_fft.set_title("DSB-SC Modulated Signal (Frequency Domain)", color="white")
        ax_mod_fft.set_xlabel("Frequency (Hz)", color="white")
        ax_mod_fft.set_ylabel("Magnitude", color="white")
        st.pyplot(fig_mod_fft)
    
    st.header("Receiver of DSB-SC")

    BWD = st.number_input("Enter BW of LPF", min_value=1, max_value=max_fc, value=1000, key="BWD")
    gain = st.number_input("Enter Gain of LPF", min_value=1, max_value=10, value=1, key="gainLPF")
    SNR = st.number_input("SNR Value with dB",min_value=0,max_value=50)
    button_demod = st.button("Show Demodulated Signal")
    if button_demod:
        demod_signal, freq_demod, mag_demod = demodulation_dsbtc(mod_signal, sr, BWD, gain, SNR)
        
        fig_demod, ax_demod = plt.subplots(figsize=(12, 4))
        ax_demod.tick_params(colors="white")
        ax_demod.set_facecolor("#222222")
        fig_demod.patch.set_facecolor("#222222")
        for spine in ax_demod.spines.values():
            spine.set_color("white")
        ax_demod.plot(np.arange(0, len(demod_signal))/sr, demod_signal, color="#00ccff")
        ax_demod.set_title("DSB-SC Demodulated Signal (Time Domain)", color="white")
        ax_demod.set_xlabel("Time (s)", color="white")
        ax_demod.set_ylabel("Amplitude", color="white")
        st.pyplot(fig_demod)
        
        fig_demod_fft, ax_demod_fft = plt.subplots(figsize=(12, 4))
        ax_demod_fft.tick_params(colors="white")
        ax_demod_fft.set_facecolor("#222222")
        fig_demod_fft.patch.set_facecolor("#222222")
        for spine in ax_demod_fft.spines.values():
            spine.set_color("white")
        ax_demod_fft.plot(freq_demod, mag_demod, color="#ff6600")
        ax_demod_fft.set_title("DSB-SC Demodulated Signal (Frequency Domain)", color="white")
        ax_demod_fft.set_xlabel("Frequency (Hz)", color="white")
        ax_demod_fft.set_ylabel("Magnitude", color="white")
        st.pyplot(fig_demod_fft)
        
        buffer = io.BytesIO()
        sf.write(buffer, demod_signal, int(sr/5), format='WAV')
        st.audio(buffer)
