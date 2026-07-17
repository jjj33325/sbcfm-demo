# SB-CFM — Scene-Conditioned Foley Sound Synthesis via Schrödinger Bridge Conditional Flow Matching

Audio demo page for our IEEE Access submission.

**Demo:** https://jjj33325.github.io/sbcfm-demo

Eojin Kim, Chanjun Chun
Department of Computer Engineering, Chosun University, Gwangju, Republic of Korea

---

## What this is

A mel-domain Foley synthesis framework for the DCASE 2023 Challenge Task 7 benchmark, which
asks a system to generate a four-second monophonic clip at 22.05 kHz from one of seven scene
categories: dog bark, footstep, gunshot, keyboard, moving motor vehicle, rain, and sneeze/cough.

The task is small-data (about 5.4 hours of labeled audio) and one-to-many: each label admits many
acoustically valid realizations, so the central tension is between scene fidelity and diversity.
Instead of reversing a diffusion chain, we transport a standard Gaussian source to the data along
an optimal-transport-coupled Brownian-bridge path and regress the closed-form velocity. The bridge
noise vanishes at both endpoints and peaks in the middle, so the marginals stay matched exactly
while the interior of each path is randomized — the model keeps the fan-out that Foley needs.

Three things distinguish the model:

- **Schrödinger-bridge path.** SB-CFM contains OT-CFM as the exact zero-noise limit, so σ is a
  continuous knob rather than a design switch. This is what makes the ablation clean.
- **RMS temporal conditioning.** The frame-level RMS envelope of the target is injected through
  block-wise FiLM, giving explicit control over *when* energy appears. It is the single most
  important component: removing it more than doubles FAD.
- **Stochastic sampler.** The deterministic Euler update is perturbed with noise scaled to the
  training bridge. The bridge and the sampler turn out to be complementary — neither alone
  accounts for the full gain.

## Results

Averaged over the seven scenes on DCASE 2023 Task 7. All numbers come from a single training run
without seed averaging.

| Model | FAD ↓ | KAD ↓ | Acc ↑ | ICD | CLAP ↑ | E-L1 ↓ |
|---|---|---|---|---|---|---|
| PixelSNAIL | 10.07 | 6.82 | 0.800 | 2.89 | 0.206 | – |
| T-Foley | 8.03 | 2.29 | 0.934 | 2.90 | 0.285 | 0.0344 |
| MambaFoley | 7.63 | 1.62 | 0.964 | 2.93 | 0.295 | 0.0374 |
| AudioLDM | 4.77 | 0.86 | 0.981 | 3.02 | 0.340 | – |
| OT-CFM (ours, σ=0) | 2.90 | 0.61 | 0.983 | 3.18 | 0.339 | 0.0232 |
| **SB-CFM (ours)** | **2.57** | **0.49** | **0.991** | 3.26 | **0.346** | **0.0230** |

ICD is a mode-collapse diagnostic, not a quantity to be maximized. E-L1 applies only to
temporally conditioned models.

Two caveats bound comparability. The AudioLDM checkpoint we run is the general-purpose model,
not the challenge entry built on it, which added task-specific pre-training on external corpora.
And our guidance scale is tuned per scene on the development split while the baselines run at
their published defaults.

## Repository layout

```
index.html                  the demo page
assets/                     figures
  pipeline.png              Fig. 1 — training and generation pipeline
  spectrograms.png          Fig. 2 — mel-spectrogram comparison, all systems x all scenes
  envelope.png              Fig. 3 — generated vs. conditioning RMS envelope, per scene
audio/                      4 s, 22.05 kHz, decoded with the official DCASE HiFi-GAN vocoder
  original/                 original recording, DCASE 2023 eval split
  pixelsnail/
  tfoley/
  mambafoley/
  audioldm/
  otcfm/                    ours, σ=0 (deterministic limit)
  sbcfm/                    ours, σ=0.05, stochastic sampler
```

Each system folder holds three clips per scene, named `<scene>_1`, `<scene>_2`, `<scene>_3`, where
`<scene>` is one of `dog_bark`, `footstep`, `gunshot`, `keyboard`, `moving_motor_vehicle`, `rain`,
`sneeze_cough`.

Within a column, every temporally conditioned system was given the same RMS envelope, taken from
the original recording in that column.

## Editing the page

The scene list, the system list, and the number of samples per scene live in a single config block
at the top of the `<script>` in `index.html`. Change those and the sample table regenerates itself.

## Citation

```bibtex
@article{kim2026sbcfm,
  title   = {Scene-Conditioned Foley Sound Synthesis via Schr\"{o}dinger Bridge Conditional Flow Matching},
  author  = {Kim, Eojin and Chun, Chanjun},
  journal = {IEEE Access},
  year    = {2026}
}
```

## Contact

jjj333@chosun.ac.kr
