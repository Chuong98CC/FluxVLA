"""PEFT adapters for the PatchWAM denoising transformer (FLUX.2 DiT).

The PatchWAM head adapts its DiT transformer blocks through the HuggingFace
``peft`` library, so any module-level PEFT tuner (LoRA, LoHa, LoKr, OFT, ...)
can be selected from configuration without new code here.
:func:`attach_dit_adapters` builds the tuner config with
``peft.get_peft_config`` and injects it with
``peft.inject_adapter_in_model``:

* the tuner replaces each target linear layer in place, keeping the released
  weights under ``<module>.base_layer`` and adding its own trainable tensors
  beside them,
* the helpers below discover the adapted modules and their trainable tensors
  generically - every tensor of an adapted module that is not inside
  ``base_layer`` - so the freeze policy, the optimizer groups and the
  checkpoint allow-list stay tuner agnostic,
* adapters are confined to the DiT block containers. The default targets are
  the block attention and feed-forward projections, and a caller-supplied
  ``target_modules`` is validated after injection, so no adapter can reach the
  denoiser's input, output, time or text projections, or anything outside the
  denoiser at all.

Prompt-learning and adaption-prompt configs do not adapt modules; they are
rejected with an actionable message instead of silently training nothing.
"""

import warnings
from collections.abc import Mapping, Sequence
from typing import Any, Optional

from torch import nn

try:
    from peft import PeftConfig, get_peft_config, inject_adapter_in_model
    from peft.tuners.tuners_utils import BaseTunerLayer
except ImportError as exc:  # pragma: no cover - exercised from a subprocess
    raise ImportError(
        'PatchWAM DiT adapters require the HuggingFace `peft` package; '
        'install it with `python -m pip install peft` (the FluxVLA base '
        'requirements pin a compatible version)') from exc

__all__ = [
    'ADAPTER_BLOCKS_ATTRIBUTE',
    'ADAPTER_CONFIG_ATTRIBUTE',
    'DIT_ADAPTER_BLOCKS',
    'DIT_ADAPTER_TARGETS',
    'RESERVED_CONFIG_KEYS',
    'adapter_block_prefixes',
    'adapter_config',
    'adapter_module_paths',
    'adapter_module_patterns',
    'adapter_parameter_names',
    'adapter_peft_type',
    'adapter_target_names',
    'attach_dit_adapters',
    'legacy_lora_config',
    'normalize_adapter_config',
]

#: Block containers searched for adapter targets. A denoiser wrapper's
#: ``transformer`` attribute is searched as well, so registered denoiser
#: wrappers work unchanged.
DIT_ADAPTER_BLOCKS: tuple = ('double_blocks', 'single_blocks')

#: Default linear layers adapted inside each DiT block. Each entry is a module
#: name suffix, resolved only inside the block containers above.
DIT_ADAPTER_TARGETS: tuple = (
    'img_attn.qkv', 'img_attn.proj', 'txt_attn.qkv', 'txt_attn.proj',
    'img_mlp.0', 'img_mlp.2', 'txt_mlp.0', 'txt_mlp.2',
    'linear1', 'linear2',
)

#: ``vla_head.peft`` keys this module owns instead of forwarding to PEFT.
RESERVED_CONFIG_KEYS: tuple = ('blocks',)

#: Legacy pre-PEFT PatchWAM adapter options, mapped one to one onto the PEFT
#: names that mean the same thing.
LEGACY_CONFIG_KEYS: dict = {
    'rank': 'r',
    'alpha': 'lora_alpha',
    'dropout': 'lora_dropout',
    'targets': 'target_modules',
}

#: The LoRA settings the deprecated ``vla_head.lora`` alias used to default to.
#: PEFT's own defaults differ (``r=8``, ``lora_alpha=8``), so an alias that
#: omits a value keeps the historical one instead of silently changing the
#: adapter it trains.
LEGACY_LORA_DEFAULTS: dict = {'r': 32, 'lora_alpha': 64, 'lora_dropout': 0.0}

#: PEFT config class used when the mapping names no ``peft_type``.
DEFAULT_PEFT_TYPE = 'LORA'

#: Attribute holding the resolved :class:`~peft.PeftConfig` on the denoiser.
ADAPTER_CONFIG_ATTRIBUTE = 'peft_adapter_config'

#: Attribute holding the dotted block-container paths on the denoiser.
ADAPTER_BLOCKS_ATTRIBUTE = 'peft_adapter_blocks'

#: Name PEFT gives the wrapped original layer inside an adapter module.
BASE_LAYER = 'base_layer'


def _is_adapter_layer(module: nn.Module) -> bool:
    return isinstance(module, BaseTunerLayer)


def _as_str(value: Any) -> str:
    return str(getattr(value, 'value', value))


def _block_containers(denoiser: nn.Module, blocks: Sequence[str]) -> list:
    """Resolve each block container as ``(dotted path, module)``.

    The dotted path names the frozen subtree from the denoiser itself
    (``transformer.double_blocks`` for the registered denoiser wrapper).
    """
    candidates = [(None, denoiser)]
    wrapped = getattr(denoiser, 'transformer', None)
    if isinstance(wrapped, nn.Module) and wrapped is not denoiser:
        candidates.append(('transformer', wrapped))
    found, seen = [], set()
    for wrapper, candidate in candidates:
        for name in blocks:
            container = getattr(candidate, name, None)
            if container is None or id(container) in seen:
                continue
            if not isinstance(container, nn.ModuleList) or not len(container):
                raise ValueError(
                    f'{name} must be a nonempty ModuleList of DiT blocks')
            found.append((name if wrapper is None else f'{wrapper}.{name}',
                          container))
            seen.add(id(container))
    if not found:
        raise ValueError(
            f'No DiT block container found; expected one of {tuple(blocks)}')
    return found


def _matches_target(name: str, target: str) -> bool:
    return name == target or name.endswith('.' + target)


def _legacy_option_names(options: Mapping[str, Any]) -> list:
    """Legacy adapter option names in use, only for a LoRA request.

    ``alpha`` and ``r`` are LoHa's own PEFT option names, so they are legacy
    spelling only when the mapping actually asks for LoRA.
    """
    requested = _as_str(options.get('peft_type', DEFAULT_PEFT_TYPE)).upper()
    if requested != DEFAULT_PEFT_TYPE:
        return []
    return sorted(key for key in options if key in LEGACY_CONFIG_KEYS)


def legacy_lora_config(config: Optional[Mapping[str, Any]] = None) -> dict:
    """Convert the deprecated ``vla_head.lora`` options to a PEFT LoRA config.

    The alias always warns: it is unconditionally deprecated, and PEFT's LoRA
    defaults are not the ones this option used to mean, so even an empty or
    partial mapping must not silently train a different adapter. Settings the
    caller omits keep their historical values (``r=32``, ``lora_alpha=64``,
    ``lora_dropout=0``, the official DiT target list) before the legacy option
    names are translated to their PEFT spelling.
    """
    options = dict(config or {})
    requested = _as_str(options.get('peft_type', DEFAULT_PEFT_TYPE)).upper()
    if requested != DEFAULT_PEFT_TYPE:
        raise ValueError(
            'The deprecated `vla_head.lora` option describes a LoRA adapter, '
            f'but peft_type is {options["peft_type"]!r}; use `vla_head.peft` '
            "with that tuner's own PEFT options instead")
    converted = {}
    for key, target in LEGACY_CONFIG_KEYS.items():
        if key not in options:
            continue
        if target in options:
            raise ValueError(
                f'The adapter config sets both the legacy {key!r} and the '
                f'PEFT {target!r} option; keep only one')
        converted[target] = options.pop(key)
    for target, default in LEGACY_LORA_DEFAULTS.items():
        converted.setdefault(target, default)
    if not converted.get('target_modules'):
        converted['target_modules'] = list(DIT_ADAPTER_TARGETS)
    options.update(converted)
    options['peft_type'] = 'LORA'
    warnings.warn(
        '`vla_head.lora` is deprecated; use `vla_head.peft='
        "dict(peft_type='LORA', r=32, lora_alpha=64, lora_dropout=0.0)` to "
        'select LoRA through the PEFT adapter library. The alias keeps the '
        f'historical LoRA defaults {LEGACY_LORA_DEFAULTS} for options it does '
        'not set.',
        DeprecationWarning,
        stacklevel=3,
    )
    return options


def normalize_adapter_config(config: Optional[Mapping[str, Any]] = None
                             ) -> dict:
    """Return a PEFT config mapping, rejecting the pre-PEFT option names.

    ``vla_head.peft`` is PEFT's own configuration surface, so the legacy LoRA
    spellings (``rank``/``alpha``/``dropout``/``targets``) are rejected with an
    actionable message rather than reinterpreted: PEFT's LoRA defaults are not
    the ones those names used to mean, so accepting them silently would change
    the adapter. A mapping with no ``peft_type`` selects LoRA, and the reserved
    ``blocks`` key stays with the caller.
    """
    options = dict(config or {})
    legacy = _legacy_option_names(options)
    if legacy:
        replacements = {key: LEGACY_CONFIG_KEYS[key] for key in legacy}
        raise ValueError(
            'The PEFT adapter config uses the legacy PatchWAM option names '
            f'{replacements}; spell them as PEFT options '
            '(r/lora_alpha/lora_dropout/target_modules), or use the '
            'deprecated `vla_head.lora` alias, which keeps the historical '
            'LoRA defaults '
            f'{LEGACY_LORA_DEFAULTS}')
    options.setdefault('peft_type', DEFAULT_PEFT_TYPE)
    return options


def _validate_target_coverage(adapted: Sequence[str],
                              target_modules: Any) -> None:
    """Require every configured target to match an adapted DiT module."""
    if isinstance(target_modules, str):
        # A single string is PEFT's regular-expression form; PEFT itself fails
        # when it matches nothing.
        return
    matched = set()
    for name in adapted:
        for target in target_modules or ():
            if _matches_target(name, target):
                matched.add(target)
    unmatched = [target for target in target_modules or ()
                 if target not in matched]
    if unmatched:
        raise ValueError(
            'PEFT target modules do not exist in the DiT blocks after '
            f'injection: {unmatched}')


def attach_dit_adapters(denoiser: nn.Module,
                        config: Optional[Mapping[str, Any]] = None) -> dict:
    """Attach PEFT adapters to the DiT blocks of ``denoiser`` in place.

    ``config`` is a PEFT configuration mapping
    (``peft_type``/``r``/``lora_alpha``/``target_modules``/...), optionally
    carrying the reserved ``blocks`` key that names the block containers to
    search. Everything is validated before the denoiser is mutated where PEFT
    allows it, and the resolved structure is validated after injection: an
    unmatched target, an adapter outside the block containers, an unsupported
    tuner type or an already adapted denoiser fails without training the wrong
    tensors.
    """
    if not isinstance(denoiser, nn.Module):
        raise TypeError('The denoiser must be an nn.Module')
    existing = adapter_module_paths(denoiser)
    if existing:
        raise ValueError(
            'The denoiser already carries PEFT adapters '
            f'({list(existing[:3])}{"..." if len(existing) > 3 else ""}); '
            'configure one adapter tuner per denoiser')
    options = normalize_adapter_config(config)
    reserved = {key: options.pop(key) for key in RESERVED_CONFIG_KEYS
                if key in options}
    blocks = reserved.get('blocks', DIT_ADAPTER_BLOCKS)
    if isinstance(blocks, str) or not isinstance(blocks, Sequence) or not all(
            isinstance(name, str) and name.strip() for name in blocks):
        raise ValueError(
            'Adapter `blocks` must be a nonempty sequence of module names')
    containers = _block_containers(denoiser, blocks)
    try:
        peft_config = get_peft_config(options)
    except KeyError as exc:
        raise ValueError(
            f'Unsupported PEFT adapter type {exc.args[0]!r}; install a `peft` '
            'version that provides it or pick a supported module-level tuner '
            "(for example 'LORA' or 'LOHA')") from exc
    except TypeError as exc:
        raise ValueError(
            f'Invalid PEFT adapter options: {exc}') from exc
    if peft_config.is_prompt_learning or peft_config.is_adaption_prompt:
        raise ValueError(
            'PatchWAM adapts DiT modules, so prompt-learning and '
            f'adaption-prompt tuners are not supported (got '
            f'{_as_str(peft_config.peft_type)}); configure a module-level '
            "tuner such as 'LORA' or 'LOHA'")
    if not getattr(peft_config, 'target_modules', None):
        options['target_modules'] = list(DIT_ADAPTER_TARGETS)
        try:
            peft_config = get_peft_config(options)
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f'Invalid PEFT adapter options: {exc}') from exc
    try:
        inject_adapter_in_model(peft_config, denoiser)
    except ValueError as exc:
        raise ValueError(
            'Could not attach the configured PEFT adapter to the PatchWAM '
            f'DiT blocks: {exc}') from exc
    adapted = adapter_module_paths(denoiser)
    if not adapted:
        raise ValueError(
            'The configured PEFT adapter matched no DiT module; check '
            f'`target_modules` against {list(DIT_ADAPTER_BLOCKS)}')
    block_paths = tuple(path for path, _ in containers)
    outside = [path for path in adapted
               if not any(path.startswith(f'{prefix}.') for prefix in
                          block_paths)]
    if outside:
        raise ValueError(
            'PEFT adapters resolved outside the DiT blocks '
            f'({outside[:4]}); the block containers are {list(block_paths)} '
            'and no adapter may touch the denoiser input, output, time or '
            'text projections')
    _validate_target_coverage(adapted, peft_config.target_modules)
    setattr(denoiser, ADAPTER_CONFIG_ATTRIBUTE, peft_config)
    setattr(denoiser, ADAPTER_BLOCKS_ATTRIBUTE, block_paths)
    return {
        'peft_type': _as_str(peft_config.peft_type),
        'config': options,
        'blocks': list(block_paths),
        'target_modules': sorted(adapted),
    }


def adapter_module_paths(model: nn.Module) -> tuple:
    """Dotted paths of the modules carrying PEFT adapters, in module order."""
    return tuple(name for name, module in model.named_modules()
                 if name and _is_adapter_layer(module))


def adapter_parameter_names(model: nn.Module) -> list:
    """Names of every PEFT adapter tensor, in module order.

    Tuner agnostic: any parameter of an adapted module that is not part of the
    wrapped ``base_layer`` belongs to the adapter (``lora_A``/``lora_B`` for
    LoRA, ``hada_w*`` for LoHa, and so on).
    """
    names = []
    for path in adapter_module_paths(model):
        prefix = f'{path}.'
        base = f'{prefix}{BASE_LAYER}.'
        for name, _ in model.get_submodule(path).named_parameters():
            full = f'{prefix}{name}'
            if not full.startswith(base):
                names.append(full)
    return names


def adapter_block_prefixes(model: nn.Module) -> tuple:
    """Dotted paths of the adapted block containers, relative to the owner.

    Returns an empty tuple when no adapters are attached, so a caller can tell
    "no adapters" from "adapted blocks that happen to resolve to no path".
    """
    for module in model.modules():
        prefixes = getattr(module, ADAPTER_BLOCKS_ATTRIBUTE, None)
        if prefixes:
            return tuple(prefixes)
    return ()


def adapter_config(model: nn.Module) -> Optional[PeftConfig]:
    """The resolved :class:`~peft.PeftConfig`, or ``None`` without adapters."""
    for module in model.modules():
        config = getattr(module, ADAPTER_CONFIG_ATTRIBUTE, None)
        if config is not None:
            return config
    return None


def adapter_peft_type(model: nn.Module) -> Optional[str]:
    """Name of the configured PEFT tuner, or ``None`` without adapters."""
    config = adapter_config(model)
    return None if config is None else _as_str(config.peft_type)


def adapter_module_patterns(model: nn.Module) -> list:
    """Compact ``<container>.*.<target>`` patterns for the adapted modules.

    One pattern per (block container, adapted target) pair instead of one per
    adapter module, so the report stays readable for the full-size FLUX.2
    transformer.
    """
    prefixes = adapter_block_prefixes(model)
    patterns = set()
    for path in adapter_module_paths(model):
        for prefix in prefixes:
            if path.startswith(f'{prefix}.'):
                _, _, suffix = path[len(prefix) + 1:].partition('.')
                patterns.add(f'{prefix}.*.{suffix}' if suffix else
                             f'{prefix}.*')
                break
        else:
            patterns.add(path)
    return sorted(patterns)


def adapter_target_names(model: nn.Module) -> dict:
    """Map each adapted module path to the config target that selected it."""
    config = adapter_config(model)
    targets = getattr(config, 'target_modules', None) if config else None
    mapping = {}
    for path in adapter_module_paths(model):
        if isinstance(targets, str):
            mapping[path] = targets
            continue
        matched = next((target for target in targets or ()
                        if _matches_target(path, target)), None)
        mapping[path] = path if matched is None else matched
    return mapping
