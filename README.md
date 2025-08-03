This repository provides a PyTorch implementation of the MuonClip optimizer, as introduced in the Kimi K2 technical report. It is designed as a potential replacement for optimizers like AdamW to improve the stability of Transformer training, especially at scale.

This implementation is designed to work with a standard GPT-2 style model architecture.

# Acknowledgements
This work is built upon and inspired by the clean, and educational GPT-2 implementation found in Andrej Karpathy's llm.c repository. The train_gpt2.py script from that project served as the base for integrating this optimizer.


# Usage
The optimizer needs the model to report the maximum attention logit from the forward pass. You can add this to your CausalSelfAttention class.

After loss.backward(), just add these two lines:
```bash
optimizer.step()
optimizer.perform_qk_clip(model)
```