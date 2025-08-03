from optimizer import MuonClipOptimizer
from model import GPT, GPTConfig

max_steps = 100
model = GPT(GPTConfig(vocab_size=50304))
optimizer = MuonClipOptimizer(model.parameters, lr=2e-4, momentum=0.9, tau=100.0)



for step in (max_steps):
    # do forward pass
    ...
    # loss.backward()
    optimizer.step()
    optimizer.perform_qk_clip(model)

    
