#!/usr/bin/env python3
# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""C-RADIO export, sample preprocessing, parity and benchmark helpers.

The C-RADIO token graph is traced at one fixed patch-aligned geometry with a
dynamic batch, its sidecar records the RADIO geometry and the graph dtypes,
and the engine is compiled through the generic :mod:`tools.trt_export`
helpers. Sample preprocessing and the parity/benchmark loops live here because
they are C-RADIO specific; the engine itself stays model-independent.
"""

import contextlib
import json
import logging
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn

from fluxvla.models.backbones.visions.base_trt import DYNAMIC_DIM
from fluxvla.models.backbones.visions.base_trt.radio_metadata import (
    DEFAULT_INPUT_NAME, DEFAULT_OUTPUT_NAME, EngineMetadata)

from . import trt_export

logger = logging.getLogger(__name__)


def restore_upstream_attention() -> Dict[str, Any]:
    """Undo the repository's global half-precision RADIO attention patch.

    Importing FluxVLA replaces ``timm``'s ``Attention.forward`` with a
    CUDA-only implementation that casts its input to bfloat16 and calls the
    installed ``flash_attn`` package. That is a deployment optimization for a
    different consumer: it cannot run in float32 and it cannot be traced, so
    an exported C-RADIO graph would be neither faithful to the upstream
    checkpoint nor compilable.

    This restores the upstream ``timm`` attention for the current process,
    which is the math the checkpoint was trained with and the math the export
    should contain. It returns a small report so a caller can record that it
    happened; call it before building or tracing the model.
    """
    import timm.models.vision_transformer as vision_transformer
    patched = vision_transformer.Attention
    injected = [
        name for name in ('inner_attn', '_flash_attn')
        if name in patched.__dict__
    ]
    upstream = _upstream_attention_forward(vision_transformer)
    # The patch replaced methods on the class object itself, so restoring them
    # in place also fixes every module that already bound that class.
    patched.forward = upstream
    for name in injected:
        delattr(patched, name)
    return {
        'restored': True,
        'still_patched': patched.forward is not upstream,
        'removed_class_attributes': injected,
    }


#: Distinguishes "the attribute was absent" from "it was present and None".
_ABSENT = object()


@contextlib.contextmanager
def upstream_attention() -> Iterator[Dict[str, Any]]:
    """Use upstream timm attention inside, the repository patch outside.

    The replacement changes class methods, so this context is for offline,
    single-threaded use only. The exact prior state is snapshotted first -
    including whether ``inner_attn`` and ``_flash_attn`` existed at all - and
    restored verbatim on exit, so an unpatched, custom-patched or patched
    process is left byte-for-byte as it was, including on an exception. No
    other model package is imported to rebuild the patch.
    """
    import timm.models.vision_transformer as vision_transformer
    attention = vision_transformer.Attention
    snapshot = {
        'forward': attention.forward,
        'inner_attn': attention.__dict__.get('inner_attn', _ABSENT),
        '_flash_attn': attention.__dict__.get('_flash_attn', _ABSENT),
    }
    report = restore_upstream_attention()
    try:
        yield report
    finally:
        attention.forward = snapshot['forward']
        for name in ('inner_attn', '_flash_attn'):
            value = snapshot[name]
            if value is _ABSENT:
                # ``type.__dict__`` is a read-only mappingproxy.
                if name in attention.__dict__:
                    delattr(attention, name)
            else:
                setattr(attention, name, value)


@contextlib.contextmanager
def strict_fp32() -> Iterator[Dict[str, Any]]:
    """Disable TF32 matmul and cuDNN kernels for an exact FP32 reference.

    PyTorch enables TF32 for cuDNN convolutions by default, which would make
    the eager reference itself a reduced-precision result and hide a real
    engine difference behind two different roundings.
    """
    matmul = torch.backends.cuda.matmul.allow_tf32
    cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield {'matmul_tf32': False, 'cudnn_tf32': False}
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul
        torch.backends.cudnn.allow_tf32 = cudnn


def _upstream_attention_forward(vision_transformer: Any) -> Any:
    """The upstream ``Attention.forward`` from the installed timm source.

    Only the class definition is re-executed, in a private namespace seeded
    with the module's own globals, so nothing is re-registered and no class
    identity changes: the export's CPE bookkeeping keeps working while the
    attention math goes back to upstream.
    """
    import ast
    import inspect
    source = Path(inspect.getfile(vision_transformer)).read_text()
    tree = ast.parse(source)
    node = next(
        item for item in tree.body
        if isinstance(item, ast.ClassDef) and item.name == 'Attention')
    namespace = dict(vars(vision_transformer))
    exec(
        compile(
            ast.Module(body=[node], type_ignores=[]),
            inspect.getfile(vision_transformer), 'exec'), namespace)
    return namespace['Attention'].__dict__['forward']


class RadioTokenExportModule(nn.Module):
    """Traceable boundary: unit-range images in, ``[B, N, D]`` tokens out.

    ``images`` are ``[B, 3, H, W]`` RGB in ``[0, 1]`` already at the encoder's
    supported geometry. ``layout='auto'`` resolves the source raster layout
    from static channel and grid axes, which is stable under tracing because
    only the batch axis is exported as dynamic.
    """

    def __init__(self, backbone: nn.Module, layout: str = 'auto') -> None:
        super().__init__()
        self.backbone = backbone
        self.layout = layout
        self.feature_dim = int(backbone.feature_dim)
        parameter = next(backbone.parameters(), None)
        self.input_dtype = (
            torch.float32 if parameter is None else parameter.dtype)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.dtype != self.input_dtype:
            images = images.to(dtype=self.input_dtype)
        return self.backbone.spatial_tokens(images, layout=self.layout)


def build_metadata(
        backbone: nn.Module,
        *,
        image_height: int,
        image_width: int,
        min_batch: int,
        opt_batch: int,
        max_batch: int,
        tensorrt_version: str,
        precision: str = 'fp32',
        input_dtype: str = 'float32',
        output_dtype: str = 'float32',
        input_name: str = DEFAULT_INPUT_NAME,
        output_name: str = DEFAULT_OUTPUT_NAME,
        source_onnx: Optional[str] = None,
        onnx_sha256: Optional[str] = None,
        external_data: Optional[str] = None,
        external_data_sha256: Optional[str] = None) -> EngineMetadata:
    """Build a validated C-RADIO record for one fixed geometry.

    ``precision`` selects builder tactics only. ``input_dtype`` and
    ``output_dtype`` describe the graph boundary and default to FP32, which is
    what the token tracer writes; they stay FP32 when an FP16 build is
    requested, because ``BuilderFlag.FP16`` does not retype network IO.
    """
    patch = int(backbone.patch_size)
    height, width = int(image_height), int(image_width)
    return EngineMetadata(
        variant=str(backbone.identifier),
        feature_dim=int(backbone.feature_dim),
        patch_size=patch,
        resolution_step=int(backbone.resolution_step),
        input_height=height,
        input_width=width,
        grid_height=height // patch,
        grid_width=width // patch,
        tensorrt_version=str(tensorrt_version),
        precision=str(precision),
        input_dtype=str(input_dtype),
        output_dtype=str(output_dtype),
        min_batch=int(min_batch),
        opt_batch=int(opt_batch),
        max_batch=int(max_batch),
        input_name=str(input_name),
        output_name=str(output_name),
        output_layout='NLD',
        source_onnx=source_onnx,
        onnx_sha256=onnx_sha256,
        external_data=external_data,
        external_data_sha256=external_data_sha256,
        created_utc=trt_export.utc_now())


def export_onnx(
    backbone: nn.Module,
    out_path: Any,
    *,
    image_height: int,
    image_width: int,
    min_batch: int = 1,
    opt_batch: int = 1,
    max_batch: int = 1,
    input_name: str = DEFAULT_INPUT_NAME,
    output_name: str = DEFAULT_OUTPUT_NAME,
    opset_version: Optional[int] = None,
    external_data: bool = True,
    external_data_min_bytes: int = 1024,
    tensorrt_version: str = '10',
    device: Any = 'cpu',
) -> Tuple[Path, EngineMetadata]:
    """Trace a PyTorch C-RADIO backbone to ONNX at one fixed geometry.

    Returns the ONNX path and the record written next to it as
    ``<out>.json``. Only the batch axis is dynamic. The graph is always FP32:
    TensorRT applies FP16 tactics at build time, which is where that decision
    belongs.
    """
    backbone = backbone.to(dtype=torch.float32)
    metadata = build_metadata(
        backbone,
        image_height=int(image_height),
        image_width=int(image_width),
        min_batch=int(min_batch),
        opt_batch=int(opt_batch),
        max_batch=int(max_batch),
        tensorrt_version=str(tensorrt_version),
        input_name=str(input_name),
        output_name=str(output_name))
    contract = metadata.contract()
    path = trt_export.export_onnx(
        RadioTokenExportModule(backbone),
        contract,
        out_path,
        opset_version=opset_version,
        external_data=external_data,
        external_data_min_bytes=int(external_data_min_bytes),
        device=device)
    data_name: Optional[str] = None
    data_hash: Optional[str] = None
    if external_data:
        _, _, references = trt_export.read_onnx_graph(path)
        if references:
            data_name = references[0]
            data_hash = trt_export.sha256_file(path.parent / data_name)
    metadata = replace(
        metadata,
        source_onnx=path.name,
        onnx_sha256=trt_export.sha256_file(path),
        external_data=data_name,
        external_data_sha256=data_hash)
    metadata.save(Path(f'{path}.json'))
    trt_export.verify_onnx_graph(
        path,
        contract,
        onnx_sha256=metadata.onnx_sha256,
        external_data=data_name,
        external_data_sha256=data_hash)
    return path, metadata


def load_image(path: Any,
               *,
               height: int = 256,
               width: int = 256) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Decode an image to one wrapper input and describe the preprocessing.

    The full image is resized directly to ``height x width`` with PIL bicubic
    (no centre crop), converted to FP32 unit RGB and then mapped to the
    wrapper's ``[-1, 1]`` endpoint range. Every backend receives the tensor
    this returns, so a comparison measures the engine, not the preprocessing.
    """
    import numpy as np
    from PIL import Image
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f'Image not found: {source}')
    with Image.open(source) as image:
        original_size = [int(value) for value in image.size]
        original_mode = str(image.mode)
        resized = image.convert('RGB').resize((int(width), int(height)),
                                              Image.BICUBIC)
    array = np.array(resized, dtype=np.uint8)
    unit = torch.from_numpy(array).to(torch.float32).permute(2, 0, 1) / 255.0
    tensor = (unit * 2.0 - 1.0).clamp_(-1.0, 1.0).unsqueeze(0).contiguous()
    report = {
        'path': str(source),
        'sha256': trt_export.sha256_file(source),
        'original_size_wh': original_size,
        'original_mode': original_mode,
        'resize_wh': [int(width), int(height)],
        'resample': 'PIL bicubic',
        'crop': None,
        'tensor_shape': [int(dim) for dim in tensor.shape],
        'tensor_dtype': str(tensor.dtype),
        'tensor_range': [-1.0, 1.0],
        'tensor_min': float(tensor.min()),
        'tensor_max': float(tensor.max()),
    }
    return tensor, report


def build_engine(
    onnx_path: Any,
    engine_path: Any,
    metadata: EngineMetadata,
    *,
    precision: str = 'fp32',
    allow_tf32: bool = False,
    workspace_bytes: int = 1 << 30,
    logger_severity: str = 'WARNING',
    verify: bool = True,
) -> Tuple[EngineMetadata, Dict[str, Any]]:
    """Compile the recorded C-RADIO graph and write its final sidecar.

    ``precision`` selects builder tactics only; the sidecar records the dtypes
    the network declares, so an FP16 build of the FP32 graph keeps a float32
    boundary. Returns the updated record and the raw build facts.
    """
    record = trt_export.build_engine(
        onnx_path,
        engine_path,
        metadata.contract(),
        precision=precision,
        allow_tf32=allow_tf32,
        workspace_bytes=int(workspace_bytes),
        logger_severity=logger_severity,
        verify=verify,
        onnx_sha256=metadata.onnx_sha256,
        external_data=metadata.external_data,
        external_data_sha256=(metadata.external_data_sha256))
    final = replace(
        metadata,
        precision=str(record['precision']),
        input_dtype=str(record['input_dtypes'][0]),
        output_dtype=str(record['output_dtypes'][0]),
        tensorrt_version=str(record['tensorrt_version']),
        workspace_bytes=int(record['workspace_bytes']),
        onnx_sha256=str(record['onnx_sha256']),
        external_data=record['external_data'],
        external_data_sha256=record['external_data_sha256'],
        created_utc=str(record['created_utc']))
    from fluxvla.models.backbones.visions.base_trt.radio_metadata import (
        default_metadata_path)
    final.save(default_metadata_path(engine_path))
    return final, record


def unit_range(images: torch.Tensor) -> torch.Tensor:
    """The exact ``[0, 1]`` tensor the traced graph consumes.

    This is the same expression ``to_unit_range`` applies, so the ONNX and
    TensorRT inputs are bit-identical to what the PyTorch reference feeds its
    encoder.
    """
    return ((images.float() + 1) / 2).clamp_(0, 1)


def prepare_sample(backbone: nn.Module,
                   image_path: Any,
                   out_dir: Any,
                   *,
                   image_height: int = 256,
                   image_width: int = 256,
                   device: Any = None) -> Dict[str, Any]:
    """Save one image's prepared inputs and the PyTorch reference features.

    The wrapper tensor, the unit-range tensor the traced graph consumes and
    the reference tokens are written to ``out_dir`` with a JSON report, so
    every later backend is compared against exactly the same input.
    """
    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    wrapper, report = load_image(
        image_path, height=image_height, width=image_width)
    unit = unit_range(wrapper).clone()
    with strict_fp32() as precision:
        reference = backbone.encode_features(
            wrapper.to(device) if device is not None else wrapper)
    report['reference_precision'] = precision
    torch.save(wrapper, out / 'input_wrapper_minus1_1.pt')
    torch.save(unit, out / 'input_unit_0_1.pt')
    torch.save(reference.tokens.detach().cpu(),
               out / 'reference_pytorch_tokens.pt')
    tokens = reference.tokens
    summary = dict(report)
    summary.update({
        'reference_shape': [int(dim) for dim in tokens.shape],
        'reference_dtype': str(tokens.dtype),
        'reference_grid': [int(value) for value in reference.grid],
        'reference_finite': bool(torch.isfinite(tokens).all()),
        'reference_min': float(tokens.min()),
        'reference_max': float(tokens.max()),
        'reference_mean': float(tokens.mean()),
        'reference_norm': float(tokens.norm()),
        'reference_requires_grad': bool(tokens.requires_grad),
        'artifacts': {
            'input_wrapper': str(out / 'input_wrapper_minus1_1.pt'),
            'input_unit': str(out / 'input_unit_0_1.pt'),
            'reference_tokens': str(out / 'reference_pytorch_tokens.pt'),
        },
    })
    (out / 'preprocess_report.json'
     ).write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    return summary


def ort_metrics(onnx_path: Any,
                unit: torch.Tensor,
                reference: torch.Tensor,
                *,
                providers: Optional[List[str]] = None,
                atol: float = 1e-4,
                rtol: float = 1e-4) -> Dict[str, Any]:
    """Run the ONNX graph under ONNX Runtime and compare it to PyTorch.

    The session receives the same unit-range tensor the PyTorch encoder sees.
    Non-CPU providers are used when requested; a requested provider that the
    installed runtime does not offer is reported rather than silently
    ignored.
    """
    import numpy as np
    import onnxruntime
    available = list(onnxruntime.get_available_providers())
    requested = list(providers or ['CPUExecutionProvider'])
    missing = [name for name in requested if name not in available]
    if missing:
        raise ValueError(f'ONNX Runtime does not offer {missing}; installed '
                         f'providers are {available}')
    session = onnxruntime.InferenceSession(str(onnx_path), providers=requested)
    feeds = {
        item.name: np.ascontiguousarray(unit.numpy())
        for item in session.get_inputs()
    }
    outputs = session.run(None, feeds)
    actual = torch.from_numpy(np.asarray(outputs[0]))
    expected = reference.detach().cpu().to(torch.float32)
    if tuple(actual.shape) != tuple(expected.shape):
        raise ValueError(
            'The ONNX graph produced shape '
            f'{tuple(actual.shape)} but the reference has '
            f'{tuple(expected.shape)}; the graph geometry does not match')
    report = feature_metrics(
        actual.float(),
        expected,
        atol=atol,
        rtol=rtol,
        runtime='onnxruntime',
        runtime_version=onnxruntime.__version__,
        providers=session.get_providers())
    report['output_names'] = [item.name for item in session.get_outputs()]
    return report


def parity_check(engine_backbone: nn.Module,
                 reference_backbone: nn.Module,
                 images: torch.Tensor,
                 *,
                 atol: float = 1e-2,
                 rtol: float = 1e-2,
                 device: Any = None) -> Dict[str, Any]:
    """Compare engine tokens with the PyTorch encoder on the same input.

    Both token sets are moved to one explicit device before a float32 upcast,
    so an fp16 engine and a bf16 encoder on different devices can be compared
    and the comparison itself is device-explicit.
    """
    engine_out = engine_backbone.encode_features(images)
    reference_out = reference_backbone.encode_features(images)
    if tuple(engine_out.grid) != tuple(reference_out.grid):
        raise ValueError(
            'Parity requires the same feature raster; the engine reports '
            f'{tuple(engine_out.grid)} and the reference '
            f'{tuple(reference_out.grid)}')
    compare_device = device
    if compare_device is None:
        compare_device = (
            engine_out.tokens.device
            if engine_out.tokens.is_cuda else reference_out.tokens.device)
    compare_device = torch.device(compare_device)
    actual = engine_out.tokens.detach().to(
        device=compare_device, dtype=torch.float32)
    expected = reference_out.tokens.detach().to(
        device=compare_device, dtype=torch.float32)
    if actual.shape != expected.shape:
        raise ValueError(
            f'Parity requires the same token shape; got {tuple(actual.shape)} '
            f'and {tuple(expected.shape)}')
    return feature_metrics(
        actual,
        expected,
        atol=atol,
        rtol=rtol,
        engine_dtype=str(engine_out.tokens.dtype),
        reference_dtype=str(reference_out.tokens.dtype),
        grid=[int(value) for value in engine_out.grid])


def feature_metrics(actual: torch.Tensor,
                    expected: torch.Tensor,
                    *,
                    atol: float = 1e-2,
                    rtol: float = 1e-2,
                    **extra: Any) -> Dict[str, Any]:
    """Numerical difference metrics between two token tensors.

    Reductions run in float64 so the reported norms and cosine are not
    themselves limited by the comparison dtype, and cosine is bounded to
    ``[-1, 1]`` because a float32 reduction can otherwise report a value
    slightly above 1 (for example 1.0000057) for an exact match. Shapes must
    match exactly: a broadcast or reshape would hide a wrong output geometry.
    """
    if tuple(actual.shape) != tuple(expected.shape):
        raise ValueError('Comparison requires identical shapes; got '
                         f'{tuple(actual.shape)} and '
                         f'{tuple(expected.shape)}')
    actual64 = actual.detach().to(torch.float64)
    expected64 = expected.detach().to(torch.float64)
    difference = (actual64 - expected64).abs()
    reference_norm = float(expected64.norm())
    relative_l2 = (
        float(difference.norm()) /
        reference_norm if reference_norm else float('nan'))
    cosine = float(
        torch.nn.functional.cosine_similarity(
            actual64.flatten(), expected64.flatten(), dim=0).clamp(-1.0, 1.0))
    report = {
        'tokens':
        int(actual.shape[1]) if actual.ndim == 3 else None,
        'feature_dim':
        int(actual.shape[-1]),
        'elements':
        int(actual.numel()),
        'max_abs_diff':
        (float(difference.max()) if difference.numel() else 0.0),
        'mean_abs_diff':
        (float(difference.mean()) if difference.numel() else 0.0),
        'relative_l2':
        relative_l2,
        'cosine':
        cosine,
        'atol':
        float(atol),
        'rtol':
        float(rtol),
        'allclose':
        bool(torch.allclose(actual64, expected64, atol=atol, rtol=rtol)),
        'finite':
        bool(torch.isfinite(actual).all()),
        'reduction_dtype':
        'float64',
    }
    report.update(extra)
    return report


def engine_parity(engine_backbone: nn.Module,
                  reference_backbone: nn.Module,
                  wrapper: torch.Tensor,
                  *,
                  atol: float = 1e-2,
                  rtol: float = 1e-2,
                  device: Any = None,
                  batch2: bool = True) -> Dict[str, Any]:
    """Sample-image parity at batch 1 and a batch-2 profile execution check."""
    report = {
        'batch1':
        parity_check(
            engine_backbone,
            reference_backbone,
            wrapper,
            atol=atol,
            rtol=rtol,
            device=device)
    }
    if batch2:
        doubled = wrapper.repeat(2, 1, 1, 1)
        engine_out = engine_backbone.encode_features(doubled)
        reference_out = reference_backbone.encode_features(doubled)
        actual = engine_out.tokens.detach().float()
        expected = reference_out.tokens.detach().to(actual.device).float()
        report['batch2'] = feature_metrics(
            actual,
            expected,
            atol=atol,
            rtol=rtol,
            shape=[int(dim) for dim in engine_out.tokens.shape],
            grid=[int(v) for v in engine_out.grid])
    return report


def benchmark_backbone(backbone: nn.Module,
                       images: torch.Tensor,
                       *,
                       warmup: int = 10,
                       iterations: int = 50,
                       device: Any = None) -> Dict[str, Any]:
    """Measure end-to-end ``encode_features`` latency on one explicit device.

    ``device`` selects the device that is synchronized and defaults to the
    input's device. Every timed iteration is followed by an explicit
    ``torch.cuda.synchronize`` on that device, so the number includes the
    preprocessing synchronization and the enqueue plus device work. These are
    measurements on this machine, not a portable speed claim.
    """
    if int(iterations) < 1:
        raise ValueError(f'iterations must be at least 1; got {iterations!r}')
    if int(warmup) < 0:
        raise ValueError(f'warmup must be non-negative; got {warmup!r}')
    if not torch.cuda.is_available():
        raise RuntimeError('Benchmarking a TensorRT engine requires CUDA')
    import statistics
    target = torch.device(device) if device is not None else images.device
    if target.type != 'cuda':
        raise ValueError(
            'Benchmarking a TensorRT engine requires a CUDA device to '
            f'synchronize; got {target}')
    if images.device != target:
        images = images.to(target)
    for _ in range(int(warmup)):
        backbone.encode_features(images)
    torch.cuda.synchronize(target)
    samples: List[float] = []
    for _ in range(int(iterations)):
        start = time.perf_counter()
        backbone.encode_features(images)
        torch.cuda.synchronize(target)
        samples.append((time.perf_counter() - start) * 1000.0)
    samples.sort()
    p90_index = min(len(samples) - 1, int(round(0.9 * (len(samples) - 1))))
    return {
        'batch': int(images.shape[0]),
        'iterations': int(iterations),
        'warmup': int(warmup),
        'mean_ms': statistics.fmean(samples),
        'median_ms': statistics.median(samples),
        'p90_ms': samples[p90_index],
        'min_ms': samples[0],
        'max_ms': samples[-1],
        'device': str(target),
        'synchronized_each_iteration': True,
    }


__all__ = [
    'DYNAMIC_DIM',
    'RadioTokenExportModule',
    'benchmark_backbone',
    'build_engine',
    'build_metadata',
    'export_onnx',
    'feature_metrics',
    'engine_parity',
    'load_image',
    'ort_metrics',
    'parity_check',
    'prepare_sample',
    'restore_upstream_attention',
    'strict_fp32',
    'unit_range',
    'upstream_attention',
]
