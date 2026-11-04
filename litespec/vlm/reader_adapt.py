"""Strict pretrained-DLite initialization and frozen-backbone integrity checks."""
import hashlib
import torch


def frozen_digest(weights):
    digest = hashlib.sha256()
    for name, value in sorted(weights.items()):
        if ".gdn_reader." in name:
            continue
        value = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def initialize_reader_adaptation(target, cfg, *, attention):
    from .baselines import load_draft
    from .gdn_reader import install_reader, reader_config
    draft, info = load_draft(cfg["pretrained_draft"], attn_implementation=attention,
                             dtype=target.dtype, output_loading_info=True)
    if any(info.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"pretrained Draft loading must be exact: {info}")
    if type(draft).__name__ != "DLiteDraftModel" or not draft.is_student:
        raise ValueError("reader adaptation requires a pretrained DLite student")
    if getattr(draft.config, "gdn_reader_config", None):
        raise ValueError("initialize from a base DLite without an existing Reader")
    text = target.config.text_config
    if draft.config.hidden_size != text.hidden_size or draft.config.vocab_size != text.vocab_size:
        raise ValueError("pretrained Draft and Target hidden/vocabulary dimensions differ")
    if draft.config.num_target_layers != text.num_hidden_layers:
        raise ValueError("Target depth differs from pretrained Draft")
    if not cfg.get("gdn_enabled") or cfg.get("algorithm", "dlite") != "dlite":
        raise ValueError("reader-only adaptation requires DLite and gdn_enabled=true")
    cfg.update(block_size=draft.block_size, num_draft_layers=len(draft.layers), chs_num_layers=draft.chs_num_layers,
               target_layer_ids=list(draft.target_layer_ids), sequential_rank=draft.sequential_rank)
    initial_digest = frozen_digest(draft.state_dict())
    draft.requires_grad_(False)
    draft.config.gdn_reader_config = reader_config(target.config, cfg)
    install_reader(draft)
    reader = draft.layers[draft.config.gdn_reader_config["injection_layer"]].gdn_reader
    reader.to(dtype=target.dtype).requires_grad_(True)
    draft.to(target.device)
    policy = dict(prefill_text="drop", prefill_vision="drop", decode_mode="none",
                  window_size=0, implementation="compact")
    draft.config.vlm_config = dict(schema_version=1, target_model=cfg["target_model"],
                                  context_policy=policy, training="frozen_reader_adaptation_v1")
    names = [name for name, p in draft.named_parameters() if p.requires_grad]
    if not names or any(".gdn_reader." not in name for name in names):
        raise RuntimeError("only Reader parameters may be trainable")
    cfg["frozen_backbone_sha256"] = initial_digest
    cfg["trainable_parameter_names"] = names
    return draft
