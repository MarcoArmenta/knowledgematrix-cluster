#!/usr/bin/env python
"""
    Tests for the single entry point to the pretrained language models,
    knowledgematrix/models/hub.py:

    - `SUPPORTED_LLMS` / `FAMILIES` / `resolve_llm`: every supported
      checkpoint has a published config fixture, and its config's
      model_type maps to the family the table claims,
    - `load_llm`: for EVERY supported checkpoint, the scaled-down
      reference model (the real transformers class on the published
      config with only its sizes reduced) goes through `load_llm` and
      comes out as the right class computing the reference's logits --
      from memory, and for one checkpoint of each kind from a directory
      on disk, where the class is picked from config.json alone,
    - `llm_footprint`: parameter counts at FULL size for every supported
      checkpoint, against counts recorded independently in the family
      test files, and the split over a device pipeline,
    - `verify_llm`: the check passes on a faithful network and fails on
      a perturbed one.

    The deep verification of each family (negative controls, knowledge
    matrix invariants, real weights) lives in llama_family.py, gemma.py
    and qwen3_5.py; their reference builders are reused here.
"""
import glob
import json
import os
import shutil
import tempfile
import unittest

import torch

from knowledgematrix.models import hub
from knowledgematrix.models.hub import (
    FAMILIES,
    SUPPORTED_LLMS,
    family_class,
    llm_class,
    llm_footprint,
    load_llm,
    resolve_llm,
)
from knowledgematrix.models.llama import LlamaLike
from knowledgematrix.models.gemma import GemmaLike
from knowledgematrix.models.qwen3_5 import Qwen3_5

try:
    import transformers
    from transformers import AutoConfig
    import safetensors  # noqa: F401
    transformers.logging.set_verbosity_error()
    # the reference builders of the three families (sibling test files)
    import llama_family as LF
    import gemma as GM
    import qwen3_5 as QW
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

torch.set_default_dtype(torch.float64)

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")

#: Dispatch check, not the precision check (the family files do that at
#: 1e-6 and below): relative to the largest reference logit. The
#: references run as shipped, i.e. with their float32 internals.
TOL = 2e-4

CLASSES = {"llama": LlamaLike, "gemma": GemmaLike, "qwen3_5": Qwen3_5}


def fixture_path(repo: str) -> str:
    name = repo.replace("/", "__") + ".json"
    for folder in ("hf_configs", "qwen3_5_siblings"):
        path = os.path.join(FIXTURES, folder, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"no config fixture for {repo}")


def published_config(repo: str):
    """The published config.json of a checkpoint, normalized by transformers."""
    folder = tempfile.mkdtemp()
    try:
        shutil.copy(fixture_path(repo), os.path.join(folder, "config.json"))
        return AutoConfig.from_pretrained(folder)
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def tiny_reference(repo: str):
    """
        (transformers model, token ids (1, T), reference logits) for the
        scaled-down copy of a supported checkpoint, built by the helpers
        of its family's test file.
    """
    family = SUPPORTED_LLMS[repo].family
    if family == "llama":
        hf, ids, outputs = LF.tiny(repo)
        return hf, ids, outputs["logits"]
    if family == "gemma":
        hf = GM.reference(repo.split("/", 1)[1])
        ids = GM.token_ids(hf)
        return hf, ids, GM.reference_logits(hf, ids)
    name = repo.split("Qwen3.5-", 1)[1]
    config = QW.scaled_config(name)
    hf = QW.build_reference(QW.Qwen3_5ForConditionalGeneration, config)
    ids = QW.token_ids(40, vocab=config.text_config.vocab_size - 8)   # below the image/video token ids
    return hf, ids, QW.reference_logits(hf, ids)


def relative_gap(reference: torch.Tensor, got: torch.Tensor) -> float:
    reference = reference.reshape(-1).double()
    got = got.reshape(-1).double()
    return ((reference - got).abs().max() / reference.abs().max()).item()


def forward(net, ids: torch.Tensor) -> torch.Tensor:
    net.eval()
    with torch.no_grad():
        return net.forward(ids.unsqueeze(0))


def headless_parameters(repo: str) -> int:
    """
        Parameters of the network WITHOUT its LM head (embedding, blocks,
        final norm) at full size, from the numbers the family test files
        record independently of this module.
    """
    family = SUPPORTED_LLMS[repo].family
    if family == "llama":
        facts = LF.CHECKPOINTS[repo]
        # transformers counts a tied table once and an untied head separately
        return facts["params"] - (0 if facts["tied"] else facts["vocab"] * facts["hidden"])
    if family == "gemma":
        return GM.PUBLISHED[repo.split("/", 1)[1]]["decoder_params"]
    return QW.PUBLISHED[repo.split("Qwen3.5-", 1)[1]][1]


class TestRegistry(unittest.TestCase):
    def test_every_supported_checkpoint_has_a_fixture(self):
        for repo in SUPPORTED_LLMS:
            with self.subTest(repo=repo):
                self.assertTrue(os.path.exists(fixture_path(repo)))

    def test_every_fixture_is_a_supported_checkpoint(self):
        paths = glob.glob(os.path.join(FIXTURES, "hf_configs", "*.json"))
        paths += glob.glob(os.path.join(FIXTURES, "qwen3_5_siblings", "*.json"))
        repos = {os.path.basename(p)[:-len(".json")].replace("__", "/") for p in paths}
        self.assertEqual(repos, set(SUPPORTED_LLMS))

    def test_the_requested_checkpoints_are_supported(self):
        for name in (
            "Qwen3.5-4B", "Qwen3.5-27B", "gemma-2-2b", "gemma-2-9b", "gemma-2-27b",
            "gemma-3-270m", "gemma-3-1b-it", "gemma-3-4b-it", "gemma-3-12b-it", "gemma-3-27b-it",
            "Qwen3-8B", "Qwen3-14B", "Llama-3.3-70B-Instruct", "SmolLM2-135M", "SmolLM2-360M",
            "SmolLM2-1.7B", "Llama-3.2-1B", "Llama-3.2-3B", "Llama-3.1-8B",
        ):
            with self.subTest(name=name):
                self.assertIn(resolve_llm(name), SUPPORTED_LLMS)

    def test_keys_are_the_repositories(self):
        for repo, llm in SUPPORTED_LLMS.items():
            self.assertEqual(repo, llm.repo)
            self.assertIn(llm.family, CLASSES)

    def test_resolve(self):
        want = "meta-llama/Llama-3.3-70B-Instruct"
        for name in ("Llama-3.3-70B-Instruct", "llama-3.3-70b-instruct", want, want.upper()):
            self.assertEqual(resolve_llm(name), want)
        self.assertEqual(resolve_llm("smollm2-135m"), "HuggingFaceTB/SmolLM2-135M")
        # anything else passes through: a mirror, a fine-tune, a directory
        self.assertEqual(resolve_llm("unsloth/gemma-2-2b"), "unsloth/gemma-2-2b")
        self.assertEqual(resolve_llm("someone/my-finetune"), "someone/my-finetune")
        folder = tempfile.mkdtemp()
        try:
            self.assertEqual(resolve_llm(folder), folder)
        finally:
            shutil.rmtree(folder)

    def test_bare_names_are_unambiguous(self):
        bare = [repo.split("/", 1)[1].lower() for repo in SUPPORTED_LLMS]
        self.assertEqual(len(bare), len(set(bare)))

    def test_family_classes(self):
        for family, cls in CLASSES.items():
            self.assertIs(family_class(family), cls)
        with self.assertRaises(ValueError):
            family_class("gpt2")

    def test_unsupported_model_type_is_refused(self):
        for model_type in ("gpt2", "qwen3_5_moe", "qwen3_next", "bert", "gemma"):
            with self.subTest(model_type=model_type):
                with self.assertRaises(ValueError):
                    llm_class(model_type)

    def test_family_table_agrees_with_the_loaders(self):
        # the Llama loader keeps its own whitelist: the two must not drift
        from knowledgematrix.models.llama import SUPPORTED_MODEL_TYPES
        self.assertEqual({t for t, f in FAMILIES.items() if f == "llama"}, set(SUPPORTED_MODEL_TYPES))

    @unittest.skipUnless(HAS_TRANSFORMERS, "requires transformers and safetensors")
    def test_published_model_types_map_to_the_claimed_family(self):
        for repo, llm in SUPPORTED_LLMS.items():
            with self.subTest(repo=repo):
                config = published_config(repo)
                self.assertIn(config.model_type, FAMILIES)
                self.assertEqual(FAMILIES[config.model_type], llm.family)
                self.assertIs(llm_class(config.model_type), CLASSES[llm.family])


@unittest.skipUnless(HAS_TRANSFORMERS, "requires transformers and safetensors")
class TestLoadLLM(unittest.TestCase):
    def test_every_supported_checkpoint_from_memory(self):
        """All supported architectures through the one entry point."""
        self.assertEqual(len(SUPPORTED_LLMS), 22)
        for repo, llm in SUPPORTED_LLMS.items():
            with self.subTest(repo=repo):
                hf, ids, reference = tiny_reference(repo)
                net = load_llm(hf, dtype=torch.float64)
                self.assertIs(type(net), CLASSES[llm.family])
                gap = relative_gap(reference, forward(net, ids))
                self.assertLess(gap, TOL, f"{repo}: logits differ by {gap:.2e} of their scale")

    def test_without_the_head_the_output_is_the_hidden_state(self):
        for repo in ("HuggingFaceTB/SmolLM2-360M", "google/gemma-2-27b", "Qwen/Qwen3.5-27B"):
            with self.subTest(repo=repo):
                hf, ids, reference = tiny_reference(repo)
                config = getattr(hf.config, "text_config", None) or hf.config
                net = load_llm(hf, include_lm_head=False, dtype=torch.float64)
                out = forward(net, ids)
                self.assertEqual(out.shape[-1], config.hidden_size)
                self.assertNotEqual(out.shape[-1], reference.shape[-1])

    def test_from_a_directory_the_class_comes_from_the_config(self):
        """One checkpoint of each kind: tied / untied, wrapper / text-only."""
        for repo in (
            "meta-llama/Llama-3.2-1B",       # llama, tied head, llama3 RoPE
            "Qwen/Qwen3-14B",                # llama family, q/k norm, untied
            "google/gemma-2-9b",             # gemma-2, final soft-cap
            "google/gemma-3-4b-it",          # gemma-3 multimodal wrapper
            "Qwen/Qwen3.5-4B",               # qwen3.5 multimodal wrapper, tied
        ):
            with self.subTest(repo=repo):
                hf, ids, reference = tiny_reference(repo)
                folder = tempfile.mkdtemp()
                try:
                    hf.save_pretrained(folder)
                    net = load_llm(folder, dtype=torch.float64)
                    self.assertIs(type(net), CLASSES[SUPPORTED_LLMS[repo].family])
                    self.assertLess(relative_gap(reference, forward(net, ids)), TOL)
                    # the default dtype of a checkpoint read from disk
                    net32 = load_llm(folder)
                    self.assertEqual({p.dtype for p in net32.parameters()}, {torch.float32})
                finally:
                    shutil.rmtree(folder, ignore_errors=True)

    def test_pipeline_and_copy_options_reach_the_loader(self):
        hf, ids, reference = tiny_reference("meta-llama/Llama-3.1-8B")
        net = load_llm(hf, dtype=torch.float64, device=["cpu", "cpu"])
        self.assertIsNotNone(net.layer_devices)
        self.assertEqual(len(net.shard_boundaries), 1)
        self.assertLess(relative_gap(reference, forward(net, ids)), TOL)
        cut = net.shard_cut_points()[2]
        net = load_llm(hf, dtype=torch.float64, device=["cpu", "cpu"], boundaries=[cut])
        self.assertEqual(net.shard_boundaries, [cut])
        shared = load_llm(hf, copy=False)
        source = hf.model.layers[0].self_attn.q_proj.weight
        ours = [m for m in shared.modules() if hasattr(m, "q_proj")][0].q_proj.weight
        self.assertEqual(ours.data_ptr(), source.data_ptr())
        self.assertTrue(load_llm(hf, save=True).save)

    def test_loader_options_reach_the_family_loader(self):
        # Gemma-2's final soft-cap is the loader's own option
        hf, ids, reference = tiny_reference("google/gemma-2-2b")
        capped = forward(load_llm(hf, dtype=torch.float64), ids)
        plain = forward(load_llm(hf, dtype=torch.float64, final_softcap=False), ids)
        self.assertLess(relative_gap(reference, capped), TOL)
        self.assertGreater(relative_gap(reference, plain), 1e-2)
        cap = hf.config.final_logit_softcapping
        self.assertLess(relative_gap(capped, cap * torch.tanh(plain / cap)), 1e-12)
        # an option no loader knows is an error, not silently dropped
        hf, ids, _ = tiny_reference("HuggingFaceTB/SmolLM2-135M")
        with self.assertRaises(TypeError):
            load_llm(hf, final_softcap=False)

    def test_unsupported_architecture_is_refused_before_any_weight_is_read(self):
        folder = tempfile.mkdtemp()
        try:
            with open(os.path.join(folder, "config.json"), "w") as f:
                json.dump({"model_type": "gpt2"}, f)
            # no weight file at all: the refusal has to come from the config
            with self.assertRaises(ValueError) as ctx:
                load_llm(folder)
            self.assertIn("gpt2", str(ctx.exception))
        finally:
            shutil.rmtree(folder)


@unittest.skipUnless(HAS_TRANSFORMERS, "requires transformers and safetensors")
class TestFootprint(unittest.TestCase):
    """Full-size sizes from the configs alone (everything on the meta device)."""

    def test_parameters_of_every_supported_checkpoint(self):
        for repo in SUPPORTED_LLMS:
            with self.subTest(repo=repo):
                config = published_config(repo)
                text = getattr(config, "text_config", None) or config
                table = text.vocab_size * text.hidden_size
                want = headless_parameters(repo)

                report = llm_footprint(config, dtype=torch.bfloat16, include_lm_head=False)
                self.assertEqual(report["class"], CLASSES[SUPPORTED_LLMS[repo].family].__name__)
                self.assertEqual(report["parameters"], want)
                self.assertEqual(report["bytes"], 2 * want)
                self.assertAlmostEqual(report["gib"], 2 * want / 2 ** 30)

                with_head = llm_footprint(config, dtype=torch.float32)
                self.assertEqual(with_head["bytes"], 4 * with_head["parameters"])
                # the head is one more (vocab, hidden) table -- except where
                # the loader shares the tied table (Qwen3.5 on one device)
                extra = with_head["parameters"] - want
                self.assertIn(extra, (0, table))
                tied_qwen = SUPPORTED_LLMS[repo].family == "qwen3_5" and config.tie_word_embeddings
                self.assertEqual(extra == 0, bool(tied_qwen))

    def test_pipeline_split(self):
        config = published_config("meta-llama/Llama-3.3-70B-Instruct")
        devices = ["cuda:0", "cuda:1", "cuda:2"]
        report = llm_footprint(config, dtype=torch.bfloat16, devices=devices, include_lm_head=False)
        stages = report["per_device"]
        self.assertEqual([s["device"] for s in stages], devices)
        self.assertEqual(sum(s["parameters"] for s in stages), report["parameters"])
        self.assertEqual(sum(s["layers"] for s in stages), report["num_layers"])
        self.assertEqual(len(report["boundaries"]), 2)
        # 69.5B parameters in bfloat16 over three devices: balanced, and
        # each stage well inside an 80 GB card
        self.assertAlmostEqual(report["gib"], 129.46, delta=0.01)
        for stage in stages:
            self.assertLess(abs(stage["gib"] - report["gib"] / 3), 2.0)
        # one device: no split is reported
        self.assertNotIn("per_device", llm_footprint(config, dtype=torch.bfloat16, devices=["cuda:0"]))

    def test_tied_head_split_over_a_pipeline_is_stored_twice(self):
        # embedding on the first device, head on the last: no sharing
        config = published_config("Qwen/Qwen3.5-4B")
        text = config.text_config
        one = llm_footprint(config, dtype=torch.bfloat16)
        two = llm_footprint(config, dtype=torch.bfloat16, devices=["cuda:0", "cuda:1"])
        self.assertEqual(two["parameters"] - one["parameters"], text.vocab_size * text.hidden_size)

    def test_accepts_a_name_a_directory_or_a_config(self):
        repo = "HuggingFaceTB/SmolLM2-135M"
        config = published_config(repo)
        want = llm_footprint(config)["parameters"]
        folder = tempfile.mkdtemp()
        try:
            shutil.copy(fixture_path(repo), os.path.join(folder, "config.json"))
            self.assertEqual(llm_footprint(folder)["parameters"], want)
        finally:
            shutil.rmtree(folder)

    def test_nothing_is_allocated(self):
        import resource
        import sys
        config = published_config("meta-llama/Llama-3.3-70B-Instruct")
        before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report = llm_footprint(config)
        after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # ru_maxrss is in bytes on macOS and in kilobytes on Linux
        unit = 1 if sys.platform == "darwin" else 1024
        self.assertGreater(report["gib"], 250)                 # 70.5B float32 parameters ...
        self.assertLess((after - before) * unit, 1 << 28)      # ... sized within 256 MB


@unittest.skipUnless(HAS_TRANSFORMERS, "requires transformers and safetensors")
class TestVerify(unittest.TestCase):
    """verify_llm on a checkpoint directory: it passes, and it can fail."""

    def _saved(self, repo):
        hf, ids, _ = tiny_reference(repo)
        folder = tempfile.mkdtemp()
        hf.save_pretrained(folder)
        self.addCleanup(shutil.rmtree, folder, True)
        return folder, ids

    def test_passes_on_each_family(self):
        from knowledgematrix.models.verify import format_report, verify_llm
        for repo in ("HuggingFaceTB/SmolLM2-135M", "google/gemma-2-2b", "google/gemma-3-1b-it",
                     "Qwen/Qwen3.5-0.8B"):
            with self.subTest(repo=repo):
                folder, ids = self._saved(repo)
                report = verify_llm(folder, input_ids=ids, dtype=torch.float32, rows=3)
                self.assertTrue(report["passed"], format_report(report))
                self.assertEqual(report["tokens"], ids.shape[1])
                self.assertEqual(report["class"], CLASSES[SUPPORTED_LLMS[repo].family].__name__)
                self.assertLess(report["relative_gap"], 1e-4)
                self.assertIn("PASS", format_report(report))

    def test_fails_when_the_network_is_not_the_reference(self):
        from knowledgematrix.models import verify
        folder, ids = self._saved("HuggingFaceTB/SmolLM2-135M")
        real_load = verify.load_llm

        def wrong_rope(source, **kwargs):
            net = real_load(source, **kwargs)
            for module in net.modules():
                if hasattr(module, "rope_theta"):
                    module.rope_theta = 10000.0        # the bug this work started from
            return net

        verify.load_llm = wrong_rope
        try:
            report = verify.verify_llm(folder, input_ids=ids, dtype=torch.float32, rows=0)
        finally:
            verify.load_llm = real_load
        self.assertFalse(report["passed"])
        self.assertGreater(report["relative_gap"], 1e-2)
        self.assertIn("FAIL", verify.format_report(report))

    def test_command_line(self):
        from knowledgematrix.models import verify
        folder, ids = self._saved("HuggingFaceTB/SmolLM2-135M")
        with self.assertRaises(SystemExit):
            verify.main(["--help"])
        self.assertEqual(verify.DTYPES["bfloat16"], torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
