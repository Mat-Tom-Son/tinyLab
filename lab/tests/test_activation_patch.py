import torch

from ..src.ablations import activation_patch


class SimpleCfg:
    def __init__(self, n_layers: int, device: str = "cpu"):
        self.n_layers = n_layers
        self.device = torch.device(device)


class FakeModel:
    """Lightweight stand-in for HookedTransformer used in unit tests."""

    def __init__(self, n_layers: int = 2, width: int = 3):
        self.cfg = SimpleCfg(n_layers)
        self.width = width
        # Minimal vocabulary for single-token conversions
        self.vocab = {
            " clean": 0,
            " corrupt": 1,
            " target": 2,
            " foil": 3,
        }
        self.vocab_size = len(self.vocab) + 2
        self.seq_len = 1

    def to(self, device):
        self.cfg.device = torch.device(device)
        return self

    def eval(self):
        return self

    def to_tokens(self, texts):
        batch = len(texts)
        tokens = torch.zeros((batch, self.seq_len), dtype=torch.long, device=self.cfg.device)
        for i, text in enumerate(texts):
            tokens[i, 0] = self.vocab.get(text, 0)
        return tokens

    def to_single_token(self, text):
        return self.vocab.get(text)

    def _layer_activation(self, layer: int, batch: int):
        return torch.full(
            (batch, self.seq_len, self.width),
            fill_value=float(layer + 1),
            device=self.cfg.device,
        )

    def run_with_cache(self, tokens):
        batch = tokens.shape[0]
        logits = torch.zeros(
            (batch, self.seq_len, self.vocab_size),
            device=self.cfg.device,
        )
        cache = {}
        for layer in range(self.cfg.n_layers):
            node = f"blocks.{layer}.hook_resid_post"
            act = self._layer_activation(layer, batch)
            cache[node] = act
            logits = logits + act.mean(dim=-1, keepdim=True)
        return logits, cache

    def run_with_hooks(self, tokens, fwd_hooks):
        batch = tokens.shape[0]
        logits = torch.zeros(
            (batch, self.seq_len, self.vocab_size),
            device=self.cfg.device,
        )
        hook_map = {node: fn for node, fn in fwd_hooks}
        for layer in range(self.cfg.n_layers):
            node = f"blocks.{layer}.hook_resid_post"
            act = self._layer_activation(layer, batch)
            if node in hook_map:
                act = hook_map[node](act.clone(), None)
            logits = logits + act.mean(dim=-1, keepdim=True)
        return logits

    def __call__(self, tokens):
        # Mirror run_with_cache logits for sham baseline.
        logits, _ = self.run_with_cache(tokens)
        return logits


def build_dataset():
    return [
        {
            "clean": " clean",
            "corrupt": " corrupt",
            "target": " target",
            "foil": " foil",
        },
        {
            "clean": " clean",
            "corrupt": " corrupt",
            "target": " target",
            "foil": " foil",
        },
    ]


def build_cfg():
    return {
        "run_name": "test_activation_patch_sham",
        "seed": 0,
        "dataset": {
            "clean_field": "clean",
            "corrupt_field": "corrupt",
            "target_field": "target",
            "foil_field": "foil",
        },
    }


def test_sham_control_is_identity():
    model = FakeModel()
    dset = build_dataset()
    cfg = build_cfg()
    battery = {
        "patch_direction": "clean->corrupt",
        "granularity": "layer_resid",
        "layers": [0, 1],
        "sham_control": True,
    }

    result = activation_patch.run(model, dset, cfg, battery, device="cpu")
    layer_table = result["layer_impact_table"]
    sham_deltas = layer_table[layer_table["metric"] == "max_abs_delta"]["value"]
    assert torch.isclose(torch.tensor(sham_deltas.values), torch.zeros_like(torch.tensor(sham_deltas.values))).all()
