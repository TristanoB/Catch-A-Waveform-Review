# Catch-A-Waveform-Review
Review and extension of the paper "Catch-A-Waveform: Learning to Generate Audio from a Single Short Example" (NeurIPS 2021) 

# Contributions 

Idea 1 : replace the multi Scale GAN with a Multi Scale diffusion model
    -> each scale being conditionned on the superior scale 

Idea 2 : adding a loss term from a self supervised pre-trained model, audio encoder 
$$
L = L_{adv} + \lambda L_{rec} + \gamma L_{SSL}
$$ 

Idea 3 : adding a forced rythmic structure for harmonicity on long range 
    -> extract the tempo and condition the generation on the rythmic structure 

Idea 4 : Multi-sample few shot learning
    -> using 3-5 similar but differente waveforms instead of only one to condition the generation 

Idea 5 : theory analysis of the stationnarity of the CAW process, mathematical formalization, formalization of CAW as an estimator of ergodic processes instead. 

Idea 6 : test the transfer of CAW trained on one instrument to another, by fixing the HF and replacing the fine scale frequencies. 

Idea 7 : Replace the convolution by a local Transformer. 
    -> as the convs have a fixed receptive field, we can use a transformer model instead. 

Idea 8 : testing on other signals, not only audio but other field time series. 

# Eval 

How to eval our new methods compared to the previous one ? 

## 1. Experimental Fairness

To ensure a valid comparison: Same training waveform (same crop, same duration), Same sampling rate (e.g., 16 kHz), Same generation length, Same number of training steps / similar compute budget, Multiple random seeds, Same normalization and preprocessing

Baselines: Original CAW, Naive cut-and-paste baseline, Our method (+ ablations)

## 2. Memorization (Copying) Metrics

We quantify whether the model copies training segments using STFT-based similarity matrices.
2.1 Copy Ratio (CR)
For each generated patch: Compute max cosine similarity with all training patches. Measure fraction above a threshold $\tau$. Lower CR → less direct copying.
2.2 Longest Copy Run (LCR)
Detect longest diagonal of high similarity. Measures length of continuous copied segment. Lower LCR → fewer sequential copy artifacts.

## 3. Mixing Analysis (Frequency-Level)

We analyze whether different frequency bands originate from different temporal positions.
3.1 Mixing Entropy (ME)
Compute alignment per frequency band. Measure entropy of source positions. Higher ME → more mixing across time.
3.2 Cross-Band Alignment Divergence (CBAD)
Compare alignment of low vs high frequency bands. Low CBAD → pure cop Moderate CBAD → structured mixin Very high CBAD → incoherent mixing

## 4. Perceptual & Objective Quality
4.1 Human Evaluation
MUSHRA-style rating (naturalness)
ABX test (real vs fake discrimination)
Preference tests
Report mean ± confidence intervals.

4.2 Objective Metrics
FAD (Fréchet Audio Distance)
Log-Spectral Distance (LSD) (for reconstruction tasks)
Embedding diversity / coverage metrics

## 5. Overfitting & Generalization
Train on crop A, evaluate similarity against crop B. Nearest-neighbor retrieval test: Distribution of minimum patch distances. Detect memorization spikes

# Useful command  

To launch the diffusion model training : 



