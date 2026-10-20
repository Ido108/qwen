"""Execute pinned samplers on tiny CPU tensors. No model files or GPU work."""
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if '--flux3' in sys.argv:
    from backend.flux3_runtime import COMFY, LIB
    sys.path.insert(0, str(COMFY))
    sys.path.insert(0, str(LIB))
else:
    sys.path.insert(0, str(ROOT / 'vendor' / 'ComfyUI'))
import comfy.options
comfy.options.enable_args_parsing(False)
from comfy.cli_args import args
args.cpu = True
import torch
import comfy.samplers
import comfy.model_sampling
from backend.res4lyf_runtime import SAMPLERS, algorithm, ensure_sampler


class Flow(comfy.model_sampling.ModelSamplingFlux, comfy.model_sampling.CONST):
    pass


class ToyGuider:
    def __init__(self):
        self.inner_model = SimpleNamespace(device=torch.device('cpu'), model_sampling=Flow(), diffusion_model=torch.nn.Identity())
        self.model_patcher = SimpleNamespace(model=self.inner_model)
        self.cfg = 3.0
        self.calls = 0

    def __call__(self, x, sigma, **kwargs):
        self.calls += 1
        return 0.2 * torch.tanh(x) + sigma.reshape((-1,) + (1,) * (x.ndim - 1)) * 0.03


def run(name, seed=123, video=False):
    model = ToyGuider()
    shape = (1, 4, 2, 4, 4) if video else (1, 4, 4, 4)
    noise = torch.randn(shape, generator=torch.Generator().manual_seed(4))
    # Comfy's prepare_noise sets the global seed before sampling. RES4LYF's
    # standard wrapper derives its SDE seed from torch.initial_seed().
    torch.manual_seed(seed)
    sigmas = comfy.samplers.calculate_sigmas(model.inner_model.model_sampling, 'beta', 5)
    events = []
    output = comfy.samplers.sampler_object(name).sample(
        model, sigmas, {'seed': seed, 'model_options': {}},
        lambda step, denoised, current, total: events.append((step, total)),
        noise, latent_image=torch.zeros_like(noise), disable_pbar=True)
    assert output.shape == noise.shape and torch.isfinite(output).all(), name
    assert events and events[-1][0] == 4 and all(total == 5 for _, total in events), (name, events)
    return output, model.calls, events


def main():
    ensure_sampler('res_2m')
    assert algorithm().__module__.endswith('.beta.rk_sampler_beta')
    outputs = {}
    for name in SAMPLERS:
        out, calls, events = run(name)
        repeat, _, _ = run(name)
        assert torch.equal(out, repeat), ('seed reproducibility', name)
        other, _, _ = run(name, seed=124)
        assert torch.equal(out, other) == name.endswith('_ode'), ('SDE/ODE seed behavior', name)
        run(name, video=True)
        outputs[name] = out
        print(f'Ready: {name}; {calls} model calls; {len(events)} progress updates; deterministic seed; finite 4D/5D CPU output', flush=True)
    assert not torch.equal(outputs['res_2m'], outputs['res_2s'])
    print('Ready: sampler execution checks passed. No model weights loaded.', flush=True)


if __name__ == '__main__':
    main()
