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
"""Offline C-RADIOv4 ONNX export and TensorRT build/validation CLI.

The pipeline is three explicit steps so each artifact can be inspected and
regenerated on its own::

    # 1. Trace a complete local official export to ONNX at one geometry.
    python tools/c_radio_tensorrt.py export-onnx \
        --radio-path /path/to/C-RADIOv4-H \
        --image-height 256 --image-width 256 \
        --min-batch 1 --opt-batch 1 --max-batch 8 \
        --out artifacts/c_radio_v4_h_256.onnx

    # 2. Compile ONNX to a serialized engine (FP32 baseline or FP16).
    python tools/c_radio_tensorrt.py build-engine \
        --onnx artifacts/c_radio_v4_h_256.onnx \
        --out artifacts/c_radio_v4_h_256.engine \
        --precision fp16 --workspace-mib 2048

    # 3. Inspect/validate the artifacts without building anything.
    python tools/c_radio_tensorrt.py inspect \
        --metadata artifacts/c_radio_v4_h_256.engine.json
    python tools/c_radio_tensorrt.py validate \
        --engine artifacts/c_radio_v4_h_256.engine
    python tools/c_radio_tensorrt.py parity \
        --engine artifacts/c_radio_v4_h_256.engine \
        --radio-path /path/to/C-RADIOv4-H

``export-onnx`` needs the optional ``onnx`` package and a complete local
official C-RADIO export; ``build-engine``, ``validate``, ``parity`` and
``benchmark`` additionally need TensorRT 10.x and a CUDA device. ``inspect``
needs neither. Nothing is downloaded, no model is substituted, and no step
is run at inference time.

``--precision`` selects builder tactics only, so the graph boundary stays
float32 in every case. ``fp32`` is the strict FP32 baseline (TF32 disabled and
precision constraints obeyed) rather than a claim of bit-exactness against the
PyTorch encoder; ``fp16`` and ``bf16`` are internal tactic requests that trade
accuracy for latency. A BF16 build needs a TensorRT release with
``BuilderFlag.BF16`` and a device with BF16 support, and it leaves FP32 in
place on the layers without BF16 kernels. It fails with a reason otherwise and
never silently becomes an FP16 build.

An FP16 or BF16 build can keep selected layers in FP32, which is the usual
remedy when a numerically sensitive operator loses accuracy at low precision::

    python tools/c_radio_tensorrt.py build-engine \
        --onnx artifacts/c_radio_v4_h_256.onnx \
        --out artifacts/c_radio_v4_h_256_fp16_ln_fp32.engine \
        --precision fp16 --precision-constraints obey \
        --layer-precision '*/LayerNormalization:fp32' --log-file build.log

``tools/trt_layer_report.py`` lists the layer names an ONNX graph produces and
counts how many each pattern matches, which is how a working pattern is chosen
instead of guessed.
"""

import argparse
import functools
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fluxvla.models.backbones.visions.base_trt import (  # noqa: E402
    radio_metadata as radio_meta)
from fluxvla.models.backbones.visions.base_trt.radio_metadata import (
    DEFAULT_INPUT_NAME, DEFAULT_OUTPUT_NAME, PRECISIONS)

from fluxvla.models.backbones.visions.radio_vit import (  # noqa: E402
    RADIO_V4_FEATURE_DIMS)
from tools import c_radio_export, trt_export  # noqa: E402


def _emit(payload: Dict[str, Any]) -> int:
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _load_metadata(
        args: argparse.Namespace,
        fallback: Optional[Path] = None) -> radio_meta.EngineMetadata:
    path = args.metadata or fallback
    if path is None:
        raise ValueError('Provide --metadata or a path with a sidecar')
    return radio_meta.EngineMetadata.load(path)


def _build_radio_backbone(radio_path: str,
                          version: str,
                          dtype: Any = None) -> Any:
    """Build the PyTorch reference encoder.

    Defaults to FP32: the ONNX graph and the engine boundary are FP32, so the
    comparison must not mix that boundary with the encoder's BF16 default.
    """
    import torch
    from fluxvla.models.backbones.visions.radio_vit import RadioViTBackbone
    return RadioViTBackbone.from_local_export(
        radio_path,
        version=version,
        feature_dim=RADIO_V4_FEATURE_DIMS[version],
        dtype=torch.float32 if dtype is None else dtype)


def _build_engine_backbone(args: argparse.Namespace) -> Any:
    from fluxvla.models.backbones.visions.radio_trt import (
        RadioViTBackboneTensorRT)
    return RadioViTBackboneTensorRT.from_engine(
        args.engine, metadata_path=args.metadata, device=args.device)


def _cuda_device(args: argparse.Namespace) -> Any:
    """Resolve the CLI device to a concrete ``cuda:<index>`` for execution."""
    import torch
    if args.device is None:
        return torch.device('cuda', torch.cuda.current_device())
    return torch.device('cuda', int(args.device))


def cmd_export_onnx(args: argparse.Namespace) -> int:
    # The traced graph must contain C-RADIO's own attention math, not the
    # repository's CUDA-only half-precision deployment patch.
    with c_radio_export.upstream_attention() as attention:
        backbone = _build_radio_backbone(args.radio_path, args.version)
        path, metadata = c_radio_export.export_onnx(
            backbone,
            args.out,
            image_height=args.image_height,
            image_width=args.image_width,
            min_batch=args.min_batch,
            opt_batch=args.opt_batch,
            max_batch=args.max_batch,
            input_name=args.input_name,
            output_name=args.output_name,
            opset_version=args.opset,
            external_data=not args.no_external_data,
            external_data_min_bytes=args.external_data_min_bytes,
            tensorrt_version=args.tensorrt_version,
            device=args.device)
    return _emit({
        'onnx': str(path),
        'metadata': metadata.to_dict(),
        'metadata_path': str(Path(f'{path}.json')),
        'attention_patch': attention,
    })


def cmd_sample(args: argparse.Namespace) -> int:
    with c_radio_export.upstream_attention() as attention:
        backbone = _build_radio_backbone(args.radio_path, args.version)
        report = c_radio_export.prepare_sample(
            backbone,
            args.image,
            args.out_dir,
            image_height=args.image_height,
            image_width=args.image_width,
            device=args.device)
    report['attention_patch'] = attention
    return _emit(report)


def cmd_build_engine(args: argparse.Namespace) -> int:
    onnx_path = Path(args.onnx)
    metadata = _load_metadata(args, fallback=Path(f'{onnx_path}.json'))
    overrides = list(args.layer_precision or [])
    # An override without a constraints mode would be a hint TensorRT may
    # ignore, so requesting one implies 'obey'; asking for either mode with no
    # override is a mistake build_engine reports.
    constraints = args.precision_constraints or ('obey'
                                                 if overrides else 'none')
    compile_engine = functools.partial(
        c_radio_export.build_engine,
        onnx_path,
        args.out,
        metadata,
        precision=args.precision,
        allow_tf32=args.allow_tf32,
        workspace_bytes=int(args.workspace_mib) << 20,
        logger_severity=args.logger_severity,
        verify=not args.no_verify,
        layer_precisions=overrides,
        precision_constraints=constraints,
        profiling_verbosity=args.profiling_verbosity)
    if args.log_file:
        # TensorRT's builder writes its diagnostics to the process streams.
        with trt_export.capture_std_streams(args.log_file):
            final, record = compile_engine()
    else:
        final, record = compile_engine()
    payload = {
        'engine': str(Path(args.out)),
        'metadata': final.to_dict(),
        'metadata_path': str(radio_meta.default_metadata_path(args.out)),
        'build': record,
    }
    if args.log_file:
        payload['log_path'] = str(Path(args.log_file))
    return _emit(payload)


def cmd_inspect(args: argparse.Namespace) -> int:
    metadata = radio_meta.EngineMetadata.load(args.metadata)
    payload = {
        'metadata': metadata.to_dict(),
        'input_shape': list(metadata.input_shape(metadata.opt_batch)),
        'output_shape': list(metadata.output_shape(metadata.opt_batch))
    }
    if args.engine:
        payload['engine'] = trt_export.load_and_validate(
            args.engine,
            metadata.contract(),
            device=args.device,
            expected_tensorrt_version=metadata.tensorrt_version)
    return _emit(payload)


def cmd_validate(args: argparse.Namespace) -> int:
    metadata = _load_metadata(
        args, fallback=radio_meta.default_metadata_path(args.engine))
    report = trt_export.load_and_validate(
        args.engine,
        metadata.contract(),
        device=args.device,
        expected_tensorrt_version=metadata.tensorrt_version)
    return _emit({'validated': True, 'report': report})


def cmd_parity(args: argparse.Namespace) -> int:
    import torch
    device = _cuda_device(args)
    args.device = device.index
    engine_backbone = _build_engine_backbone(args)
    try:
        with c_radio_export.upstream_attention():
            # The reference runs on the engine's device in FP32, the dtype the
            # engine boundary was exported in.
            reference = _build_radio_backbone(args.radio_path,
                                              args.version).to(device)
            if args.image or args.artifacts:
                wrapper = c_radio_export.load_image(
                    args.image or Path(args.artifacts) /
                    'input_wrapper_minus1_1.pt',
                    height=engine_backbone.metadata.input_height,
                    width=engine_backbone.metadata.input_width)[0] \
                    if args.image else torch.load(
                        Path(args.artifacts) / 'input_wrapper_minus1_1.pt')
                with c_radio_export.strict_fp32():
                    report = c_radio_export.engine_parity(
                        engine_backbone,
                        reference,
                        wrapper,
                        atol=args.atol,
                        rtol=args.rtol,
                        device=device,
                        batch2=not args.no_batch2)
            else:
                with c_radio_export.strict_fp32():
                    metadata = engine_backbone.metadata
                    generator = torch.Generator(device='cpu').manual_seed(
                        args.seed)
                    images = torch.rand(
                        metadata.opt_batch,
                        3,
                        *metadata.image_size,
                        generator=generator).to(device) * 2 - 1
                    report = {
                        'batch1':
                        c_radio_export.parity_check(
                            engine_backbone,
                            reference,
                            images,
                            atol=args.atol,
                            rtol=args.rtol,
                            device=device)
                    }
    finally:
        engine_backbone.close()
    passed = all(part['allclose'] for part in report.values()
                 if isinstance(part, dict) and 'allclose' in part)
    report['passed'] = passed
    if args.out:
        Path(args.out).write_text(
            __import__('json').dumps(report, indent=2, sort_keys=True) + '\n')
    _emit(report)
    # A numerical mismatch is a real failure, not a successful run.
    return 0 if passed else 1


def cmd_ort_parity(args: argparse.Namespace) -> int:
    import torch
    artifacts = Path(args.artifacts)
    unit = torch.load(artifacts / 'input_unit_0_1.pt')
    reference = torch.load(artifacts / 'reference_pytorch_tokens.pt')
    report = c_radio_export.ort_metrics(
        args.onnx,
        unit,
        reference,
        providers=args.providers,
        atol=args.atol,
        rtol=args.rtol)
    if args.out:
        Path(args.out).write_text(
            __import__('json').dumps(report, indent=2, sort_keys=True) + '\n')
    _emit(report)
    return 0 if report['allclose'] else 1


def cmd_benchmark(args: argparse.Namespace) -> int:
    import torch
    device = _cuda_device(args)
    args.device = device.index
    engine_backbone = _build_engine_backbone(args)
    metadata = engine_backbone.metadata
    generator = torch.Generator(device='cpu').manual_seed(0)
    images = torch.rand(
        metadata.opt_batch, 3, *metadata.image_size,
        generator=generator).to(device) * 2 - 1
    try:
        report = c_radio_export.benchmark_backbone(
            engine_backbone,
            images,
            device=device,
            warmup=args.warmup,
            iterations=args.iterations)
    finally:
        engine_backbone.close()
    report['precision'] = metadata.precision
    report['note'] = ('Measured on this device only; the encoder path is not '
                      'benchmarked here, so this is not a speedup claim.')
    return _emit(report)


def _add_engine_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        '--engine', required=True, help='Serialized TensorRT .engine file.')
    parser.add_argument(
        '--metadata',
        default=None,
        help='Metadata sidecar (default: <engine>.json).')
    parser.add_argument(
        '--device',
        type=int,
        default=None,
        help='CUDA device index (default: current device).')


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser without loading any model or engine."""
    parser = argparse.ArgumentParser(
        prog='c_radio_tensorrt',
        description='Offline C-RADIOv4 ONNX export and TensorRT 10.x build.')
    sub = parser.add_subparsers(dest='command', required=True)

    export = sub.add_parser(
        'export-onnx', help='Trace a local C-RADIO export to ONNX.')
    export.add_argument(
        '--radio-path',
        required=True,
        help='Complete local official C-RADIO export.')
    export.add_argument(
        '--version',
        default='c-radio_v4-h',
        choices=sorted(RADIO_V4_FEATURE_DIMS))
    export.add_argument('--image-height', type=int, required=True)
    export.add_argument('--image-width', type=int, required=True)
    export.add_argument('--out', required=True, help='Output .onnx path.')
    export.add_argument('--min-batch', type=int, default=1)
    export.add_argument('--opt-batch', type=int, default=1)
    export.add_argument('--max-batch', type=int, default=1)
    export.add_argument('--input-name', default=DEFAULT_INPUT_NAME)
    export.add_argument('--output-name', default=DEFAULT_OUTPUT_NAME)
    export.add_argument('--opset', type=int, default=None)
    export.add_argument(
        '--no-external-data',
        action='store_true',
        help='Keep all weights inline in the .onnx file.')
    export.add_argument('--external-data-min-bytes', type=int, default=1024)
    export.add_argument(
        '--tensorrt-version',
        default='10',
        help='Target TensorRT major recorded in the sidecar.')
    export.add_argument('--device', default='cpu')
    export.set_defaults(handler=cmd_export_onnx)

    sample = sub.add_parser(
        'sample', help='Prepare one image and its PyTorch reference tokens.')
    sample.add_argument('--radio-path', required=True)
    sample.add_argument(
        '--version',
        default='c-radio_v4-h',
        choices=sorted(RADIO_V4_FEATURE_DIMS))
    sample.add_argument('--image', required=True, help='Input image file.')
    sample.add_argument('--out-dir', required=True)
    sample.add_argument('--image-height', type=int, default=256)
    sample.add_argument('--image-width', type=int, default=256)
    sample.add_argument('--device', default=None)
    sample.set_defaults(handler=cmd_sample)

    ort = sub.add_parser(
        'ort-parity',
        help='Compare an ONNX graph with the saved reference '
        'under ONNX Runtime.')
    ort.add_argument('--onnx', required=True)
    ort.add_argument(
        '--artifacts',
        required=True,
        help='Directory written by the sample command.')
    ort.add_argument('--out', default=None)
    ort.add_argument('--providers', nargs='*', default=None)
    ort.add_argument('--atol', type=float, default=1e-4)
    ort.add_argument('--rtol', type=float, default=1e-4)
    ort.set_defaults(handler=cmd_ort_parity)

    build = sub.add_parser(
        'build-engine', help='Compile ONNX to a serialized TensorRT engine.')
    build.add_argument('--onnx', required=True)
    build.add_argument('--out', required=True, help='Output .engine path.')
    build.add_argument(
        '--metadata',
        default=None,
        help='Metadata sidecar (default: <onnx>.json).')
    build.add_argument(
        '--precision',
        default='fp32',
        choices=list(PRECISIONS),
        help='Builder tactics only (BuilderFlag.FP16 or BF16); it does not '
        'change the graph input/output dtypes.')
    build.add_argument('--workspace-mib', type=int, default=1024)
    build.add_argument(
        '--allow-tf32',
        action='store_true',
        help='Allow TF32 tactics in an FP32 build (off by default so an FP32 '
        'build keeps FP32 tactics instead of silently running TF32).')
    build.add_argument(
        '--layer-precision',
        action='append',
        type=trt_export.parse_layer_precision,
        default=None,
        metavar='PATTERN:DTYPE',
        help='Repeatable per-layer precision constraint, matched against '
        'layer names with *. For example '
        '--layer-precision "*/LayerNormalization:fp32" keeps every '
        'LayerNormalization in FP32 in an FP16 build. Defaults to '
        '--precision-constraints obey, which fails rather than overriding '
        'the constraint; a pattern that matches no layer fails the build. '
        'List the layer names with tools/trt_layer_report.py.')
    build.add_argument(
        '--precision-constraints',
        choices=list(trt_export.PRECISION_CONSTRAINTS),
        default=None,
        help='What a --layer-precision constraint is worth: obey (require '
        'it, failing rather than overriding), prefer (ask for it and allow '
        'TensorRT\'s documented fallback), or none (a hint TensorRT may '
        'ignore). Defaults to obey when a layer precision is requested, '
        'else none.')
    build.add_argument(
        '--profiling-verbosity',
        choices=list(trt_export.PROFILING_LEVELS),
        default=None,
        help='Per-layer detail the engine keeps for the inspector. detailed '
        'is what makes a per-layer datatype inspection of the built engine '
        'possible (default: TensorRT\'s own default, layer_names_only).')
    build.add_argument(
        '--log-file',
        default=None,
        help='Append the builder\'s own diagnostics (its precision warnings '
        'and tactic notes) to this file.')
    build.add_argument('--logger-severity', default='WARNING')
    build.add_argument(
        '--no-verify',
        action='store_true',
        help='Skip deserializing the freshly built engine.')
    build.set_defaults(handler=cmd_build_engine)

    inspect = sub.add_parser(
        'inspect', help='Print a metadata sidecar (no TensorRT).')
    inspect.add_argument('--metadata', required=True)
    inspect.add_argument(
        '--engine',
        default=None,
        help='Optionally also load this engine and validate.')
    inspect.add_argument('--device', type=int, default=None)
    inspect.set_defaults(handler=cmd_inspect)

    validate = sub.add_parser(
        'validate', help='Deserialize an engine and validate metadata vs IO.')
    _add_engine_arguments(validate)
    validate.set_defaults(handler=cmd_validate)

    parity = sub.add_parser(
        'parity', help='Compare engine tokens with the PyTorch encoder.')
    _add_engine_arguments(parity)
    parity.add_argument('--radio-path', required=True)
    parity.add_argument(
        '--version',
        default='c-radio_v4-h',
        choices=sorted(RADIO_V4_FEATURE_DIMS))
    parity.add_argument('--atol', type=float, default=1e-2)
    parity.add_argument('--rtol', type=float, default=1e-2)
    parity.add_argument('--seed', type=int, default=0)
    parity.add_argument(
        '--image',
        default=None,
        help='Compare on this image instead of noise.')
    parity.add_argument(
        '--artifacts',
        default=None,
        help='Directory written by the sample command.')
    parity.add_argument(
        '--out', default=None, help='Write the parity report JSON here.')
    parity.add_argument(
        '--no-batch2',
        action='store_true',
        help='Skip the batch-2 profile execution check.')
    parity.set_defaults(handler=cmd_parity)

    bench = sub.add_parser(
        'benchmark', help='Measure engine encode latency on this device.')
    _add_engine_arguments(bench)
    bench.add_argument('--warmup', type=int, default=10)
    bench.add_argument('--iterations', type=int, default=50)
    bench.set_defaults(handler=cmd_benchmark)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except Exception as error:  # noqa: BLE001 - CLI reports, never traces
        print(f'error: {type(error).__name__}: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
