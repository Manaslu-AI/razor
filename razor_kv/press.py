"""RAZOR press for kvpress: same gate architecture as KVzap (per-layer MLP on hidden states,
one score per KV head), loaded from a local checkpoint. Only the training objective differs."""
from dataclasses import dataclass

from kvpress.presses.kvzap_press import KVzapModel, KVzapPress


@dataclass
class LocalGatePress(KVzapPress):
    path: str = ""

    def post_init_from_model(self, model):
        if getattr(self, "_loaded", None) != self.path:
            self.kvzap_model = KVzapModel.from_pretrained(self.path)
            self._loaded = self.path
