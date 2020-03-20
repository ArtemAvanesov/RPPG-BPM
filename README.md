# RPPG-BPM
This is a project for diploma (graduate work)
It is dedicated to calculating heart rate value from a video recording with a person’s face

The input data are the average values of RGB-channels of face area

The pulse calculation procedure consists of two steps:
1) calculating a one-dimensional rppg-signal that contains a pulse
2) calculation of beats per minute

Methods for first stage:
- CHROM (Chrominance-based method)
- POS (Plane-Orthogonal-to-Skin method)
- ICA (Independent Component Analysis)

Methods for second stage:
- interbeats analysis
- fourier analysis
- wavelet analysis

Required libraries:
- numpy
- matplotlib
- pandas
- scipy
- sklearn
- PyWavelets

Releted works:
1.	G. de Haan and V. Jeanne, “Robust pulse rate from chrominance-based rppg”, Biomedical Engineering, IEEE Transactions on, vol. 60, no. 10, pp. 2878–2886, Oct. 2013.
2.	W. Wang, AC den Brinker, S. Stuijk, G. de Haan, “Algorithmic principles of remote PPG”, Biomedical Engineering, IEEE Transactions on, vol. 64, no. 7, pp. 1479–1491, July 2017.
3.	W. Wang, S. Stuijk, G. de Haan “A novel algorithm for remote photoplethysmography: Spatial subspace rotation”, Biomedical Engineering, IEEE Transactions on, vol. 63, no. 9, pp. 1974–1984, Sept. 2016.
4.	Pavithra Solai Jawahar, “Pulse extraction using POS algorithm” / GitHub: https://github.com/pavisj/rppg-pos/blob/master/pos_face_seg.py
