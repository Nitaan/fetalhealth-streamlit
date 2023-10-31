import pickle
import numpy as np
import streamlit as st
from sklearn import tree


# load save model  
model = pickle.load((open('fetal_model.sav', 'rb')))

st.image("header.png")
# judul web
st.title('Klasifikasi Kesehatan Janin')

col1, col2, col3 = st.columns(3)
    
with col1:
    baseline_value = st.slider ('Input Garis dasar detak jantung janin (denyut per menit)', value=120.0, 110, 160)
with col1:
    accelerations = st.number_input ('Input Jumlah percepatan per detik', value=0.0, step=0.001, format="%0.3f")	
with col1:
    fetal_movement = st.number_input ('Input Jumlah gerakan janin per detik', value=0.0, step=0.001, format="%0.3f")	
with col1:
    uterine_contractions = st.number_input ('Input Jumlah kontraksi uterus per detik', value=0.0, step=0.001, format="%0.3f")		
with col1:
    light_decelerations = st.number_input ('Input Jumlah perlambatan ringan per detik', value=0.0, step=0.001, format="%0.3f")
with col1:
    severe_decelerations = st.number_input ('Input 	Jumlah perlambatan parah per detik', value=0.0, step=0.001, format="%0.3f")
with col1:
    prolongued_decelerations = st.number_input ('Input Jumlah perlambatan berkepanjangan per detik', value=0.0, step=0.001, format="%0.3f")
with col2:
    abnormal_short_term_variability	= st.slider ('Input Persentase waktu dengan variabilitas jangka pendek yang abnormal', min_value=0,
    max_value=100, value=73.0, step=0.001,  format="%d%%")
with col2:
    mean_value_of_short_term_variability = st.number_input ('Input Nilai rata-rata variabilitas jangka pendek', value=0.5, step=0.001, format="%0.3f")
with col2:
    percentage_of_time_with_abnormal_long_term_variability = st.number_input ('Input Persentase waktu dengan variabilitas jangka panjang yang abnormal', min_value=0,
    max_value=100, value=43.0, step=0.001,  format="%d%%")
with col2:
    mean_value_of_long_term_variability = st.number_input ('Input Nilai rata-rata variabilitas jangka panjang', value=2.4, step=0.001, format="%0.3f")
with col2:
    histogram_width = st.number_input ('Input Lebar histogram FHR', value=64.0, step=0.001, format="%0.3f")
with col2:
    histogram_min = st.number_input ('Input Histogram FHR Minimum', value=62.0, step=0.001, format="%0.3f")
with col2:
    histogram_max = st.number_input ('Input Histogram FHR Maksimum', value=126.0, step=0.001, format="%0.3f")
with col3:
    histogram_number_of_peaks = st.number_input ('Input Jumlah Puncak Histogram',value=2.0,  step=0.001, format="%0.3f")
with col3:
    histogram_number_of_zeroes = st.number_input ('Input Jumlah histogram nol',value=0.0,  step=0.001, format="%0.3f")
with col3:
    histogram_mode = st.number_input ('Input Modus histogram', value=120.0, step=0.001, format="%0.3f")
with col3:
    histogram_mean = st.number_input ('Input Mean histogram', value=137.0, step=0.001, format="%0.3f")
with col3:
    histogram_median = st.number_input ('Input Median histogram', value=121.0, step=0.001, format="%0.3f")
with col3:
    histogram_variance = st.number_input ('Input Varians histogram', value=73.0, step=0.001, format="%0.3f")
with col3:
    histogram_tendency = st.number_input ('Input Kecenderungan histogram', value=1.0, step=0.001, format="%0.3f")

# code for prediction
fetal_health = ''

if st.button("Test Prediksi Kesehatan Janin"):
    fetal_prediction = model.predict([[baseline_value,accelerations,fetal_movement,uterine_contractions,
                        light_decelerations,severe_decelerations,prolongued_decelerations,
                        abnormal_short_term_variability,mean_value_of_short_term_variability,
                        percentage_of_time_with_abnormal_long_term_variability,
                        mean_value_of_long_term_variability,histogram_width,histogram_min,
                        histogram_max,histogram_number_of_peaks,histogram_number_of_zeroes,
                        histogram_mode,histogram_mean,histogram_median,histogram_variance,histogram_tendency]])
    
    if (fetal_prediction[0] == 1):
        fetal_health = "Pasien Normal"
    elif (fetal_prediction[0] == 2):
        fetal_health = 'Pasien Suspect'
    else:
        fetal_health = 'Pasien Pathological'
st.success(fetal_health)
