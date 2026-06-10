# =============================================================================
# Transition-aware KD — dev image
#
# Base: NVIDIA NeMo container. The PyTorch inside is built for Blackwell
#       (sm_120 / RTX 5090). nemo:25.07 (NeMo 2.4.0rc3) is the documented
#       known-good floor for sm_120 — older tags (e.g. 24.xx) FAIL on RTX 5090.
#       Use a newer tag here if one is available.
#
# Pull may require an NGC login:  docker login nvcr.io   (free NGC account)
# =============================================================================
FROM nvcr.io/nvidia/nemo:25.07

# Project root inside the container.
# At run time you mount your working copy over this for live editing (see README).
WORKDIR /workspace/transition_aware_kd

# Extra Python deps beyond the NeMo base (torch, nemo_toolkit, lightning,
# hydra, omegaconf, etc. are ALREADY in the base image — don't reinstall them).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source for a self-contained/reproducible image.
# During active dev the mount (-v) shadows this with your live files.
COPY . .

CMD ["/bin/bash"]