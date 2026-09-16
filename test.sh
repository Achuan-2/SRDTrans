#!/usr/bin/env bash
set -e
# Run from a terminal with the srdtrans environment already activated.
python test.py \
  --datasets_path datasets \
  --datasets_folder PFC \
  --denoise_model PFC_202609160005 \
  --GPU 0 \
  --patch_x 64 \
  --patch_t 160
