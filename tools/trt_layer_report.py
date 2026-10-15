#!/usr/bin/env python3
"""List the layers of an ONNX graph or of a built TensorRT engine.

This answers the two questions that come up before a layer precision override
can be written and believed:

* Which layer names does this ONNX graph produce, and how many of them does a
  pattern match? Layer names come from the ONNX node names and a pattern is
  matched with ``*``/``?`` wildcards, exactly like ``trtexec
  --layerPrecisions``. This is how a pattern is chosen instead of guessed.
* Which datatype did each layer of an already-built engine end up in? That is
  only answerable when the engine was built with ``--profiling-verbosity
  detailed``; an engine built with the default verbosity reports layer names
  without datatypes, and this tool says so rather than inventing an answer.
  The ``LayerType`` an engine reports is the implementation TensorRT selected
  (``CaskConvolution``, say), not the network layer type of the graph.

The tool is model independent: it needs the graph or the serialized engine, not
the model that produced them. It parses and inspects only - it never builds an
engine, and it writes nothing unless ``--out`` is given.

    python tools/trt_layer_report.py --onnx artifacts/c_radio_v4_h_256.onnx \
        --match '*/LayerNormalization' --out layers.json
    python tools/trt_layer_report.py --engine artifacts/model.engine
"""

import argparse
import fnmatch
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fluxvla.models.backbones.visions.base_trt import (  # noqa: E402
    require_tensorrt)
from tools import trt_export  # noqa: E402

#: Field names the engine inspector uses in its JSON layer description.
_LAYER_NAME_FIELD = 'Name'
_LAYER_TYPE_FIELD = 'LayerType'
_DATATYPE_FIELD = 'Format/Datatype'

#: Fallback for a TensorRT build whose inspector emits non-JSON layer text.
_DATATYPE_PATTERN = re.compile(r'"Format/Datatype":\s*"(\w+)"')
_LAYER_TYPE_PATTERN = re.compile(r'"LayerType":\s*"(\w+)"')
_LAYER_NAME_PATTERN = re.compile(r'"Name":\s*"([^"]*)"')

#: Layer names a report keeps per pattern.
_NAME_LIMIT = 64


class _ParsedGraph(NamedTuple):
    """A parsed network plus the TensorRT objects it points into.

    TensorRT's network keeps references into the logger, builder and parser
    that created it, so all four have to stay alive together for as long as
    anything reads the network.
    """

    logger: Any
    builder: Any
    network: Any
    parser: Any


def _parse_onnx_network(trt: Any, onnx_path: Path) -> _ParsedGraph:
    """Parse an ONNX graph into a TensorRT network without building."""
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = trt_export.create_network(builder, trt)
    parser = trt.OnnxParser(network, logger)
    parse_from_file = getattr(parser, 'parse_from_file', None)
    ok = (
        bool(parse_from_file(str(onnx_path))) if parse_from_file is not None
        else bool(parser.parse(onnx_path.read_bytes())))
    if not ok:
        errors = [
            str(parser.get_error(index))
            for index in range(int(getattr(parser, 'num_errors', 0)))
        ]
        raise RuntimeError(f'TensorRT could not parse {onnx_path}: ' + (
            '; '.join(errors) if errors else 'no parser error was reported'))
    return _ParsedGraph(
        logger=logger, builder=builder, network=network, parser=parser)


def describe_graph(onnx_path: Any,
                   *,
                   patterns: Optional[Sequence[str]] = None,
                   name_limit: int = _NAME_LIMIT) -> Dict[str, Any]:
    """Layer names and types of an ONNX graph, plus pattern match counts.

    A pattern that matches nothing is reported with ``matched_layers: 0``
    rather than raising, because checking a candidate pattern before using it
    is the point; a zero count is the answer, not a failure.
    """
    trt = require_tensorrt()
    path = Path(onnx_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f'ONNX model not found: {path}')
    graph = _parse_onnx_network(trt, path)
    # ``graph`` stays bound for the whole read: the network points into the
    # parser and the logger it was parsed with.
    network = graph.network
    layers: List[Dict[str, Any]] = []
    for index in range(int(network.num_layers)):
        layer = network.get_layer(index)
        layers.append({
            'index':
            index,
            'name':
            str(layer.name),
            'type':
            trt_export._layer_type_name(layer),
            'output_dtypes': [
                str(layer.get_output(slot).dtype)
                for slot in range(int(layer.num_outputs))
            ],
        })
    matches: List[Dict[str, Any]] = []
    for pattern in patterns or []:
        found = [entry for entry in layers if _matches(entry['name'], pattern)]
        matches.append({
            'pattern':
            pattern,
            'matched_layers':
            len(found),
            'layer_types':
            sorted({entry['type']
                    for entry in found}),
            'layer_names':
            [entry['name'] for entry in found][:int(name_limit)],
            'truncated':
            len(found) > int(name_limit),
        })
    return {
        'onnx':
        str(path),
        'onnx_sha256':
        trt_export.sha256_file(path),
        'tensorrt_version':
        str(trt.__version__),
        'network_layers':
        len(layers),
        'layer_type_counts':
        dict(sorted(Counter(entry['type'] for entry in layers).items())),
        'pattern_matches':
        matches,
        'layers':
        layers,
    }


def _matches(name: str, pattern: str) -> bool:
    return fnmatch.fnmatchcase(name, pattern)


def _first_match(pattern: Any, text: str) -> str:
    found = pattern.search(text)
    return '' if found is None else str(found.group(1))


def _tensor_datatypes(entries: Any) -> List[str]:
    """The ``Format/Datatype`` markers of a layer's input or output tensors."""
    found: List[str] = []
    for entry in entries or []:
        if isinstance(entry, dict) and entry.get(_DATATYPE_FIELD):
            marker = str(entry[_DATATYPE_FIELD])
            if marker not in found:
                found.append(marker)
    return found


def _named_layer(
    name: Any,
    layer_type: Any = '',
    datatypes: Sequence[str] = (),
    inputs: Sequence[str] = (),
    outputs: Sequence[str] = ()
) -> Dict[str, Any]:
    """A layer description; any side that is unknown stays empty."""
    inputs = [str(value) for value in inputs]
    outputs = [str(value) for value in outputs]
    every = {str(value) for value in datatypes} | set(inputs) | set(outputs)
    return {
        'name': str(name),
        'type': str(layer_type),
        'input_datatypes': inputs,
        'output_datatypes': outputs,
        'datatypes': sorted(every),
    }


def _layer_information(info: Any) -> Dict[str, Any]:
    """Field-shaped view of one engine inspector layer description.

    The inspector's answer depends on the verbosity the engine was built with.
    A ``DETAILED`` engine returns a JSON object whose ``Inputs``/``Outputs``
    entries carry a ``Format/Datatype`` marker, and whose ``LayerType`` is the
    *implementation* TensorRT selected (``CaskConvolution``, say) rather than
    the network layer type the ONNX parser reported. An engine built with the
    default verbosity returns one bare layer name and nothing else.
    """
    text = str(info)
    try:
        parsed: Any = json.loads(text)
    except ValueError:
        parsed = None
    if isinstance(parsed, dict):
        return _named_layer(
            parsed.get(_LAYER_NAME_FIELD, ''),
            parsed.get(_LAYER_TYPE_FIELD, ''),
            inputs=_tensor_datatypes(parsed.get('Inputs')),
            outputs=_tensor_datatypes(parsed.get('Outputs')))
    if isinstance(parsed, str):
        return _named_layer(parsed)
    if _LAYER_NAME_PATTERN.search(text) or _DATATYPE_PATTERN.search(text):
        # Not JSON, but inspector-shaped: read the fields out of the text.
        return _named_layer(
            _first_match(_LAYER_NAME_PATTERN, text),
            _first_match(_LAYER_TYPE_PATTERN, text),
            datatypes=sorted(set(_DATATYPE_PATTERN.findall(text))))
    # Anything else is a bare layer name.
    return _named_layer(text.strip())


def describe_engine(engine_path: Any) -> Dict[str, Any]:
    """Per-layer datatypes of a built engine, as the inspector reports them."""
    trt = require_tensorrt()
    path = Path(engine_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f'Engine not found: {path}')
    serialized = path.read_bytes()
    # The engine points into the runtime and the logger, so all three stay
    # bound until the inspector has finished reading the layers.
    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized)
    if engine is None:
        raise RuntimeError(
            f'TensorRT could not deserialize {path}; it was built by a '
            'different TensorRT version, CUDA/driver or GPU architecture')
    layers: List[Dict[str, Any]] = []
    inspector = engine.create_engine_inspector()
    for index in range(int(engine.num_layers)):
        layers.append(
            _layer_information(
                inspector.get_layer_information(
                    index, trt.LayerInformationFormat.JSON)))
    mentions = Counter()
    described = 0
    for entry in layers:
        for name in entry['datatypes']:
            mentions[name] += 1
        if entry['type']:
            described += 1
    return {
        'engine':
        str(path),
        'engine_bytes':
        len(serialized),
        'tensorrt_version':
        str(trt.__version__),
        'engine_layers':
        len(layers),
        'datatype_mentions':
        dict(sorted(mentions.items())),
        'datatype_is_available':
        bool(mentions),
        'layers_with_a_description':
        described,
        'note': ('Per-layer datatypes need an engine built with '
                 '--profiling-verbosity detailed; this engine reports names '
                 'only.' if not mentions else
                 'Datatypes and layer types come from the engine inspector; '
                 'LayerType names the implementation TensorRT selected, not '
                 'the network layer type in the ONNX graph.'),
        'layer_type_counts':
        dict(
            sorted(
                Counter(entry['type'] for entry in layers
                        if entry['type']).items())),
        'layers':
        layers,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='trt_layer_report',
        description='List ONNX graph layers or engine layer datatypes.')
    parser.add_argument(
        '--onnx',
        default=None,
        help='ONNX graph to parse and list (nothing is built).')
    parser.add_argument(
        '--engine',
        default=None,
        help='Serialized engine to inspect (needs --profiling-verbosity '
        'detailed at build time for datatypes).')
    parser.add_argument(
        '--match',
        action='append',
        default=None,
        metavar='PATTERN',
        help='Repeatable layer-name pattern to count, using * wildcards. '
        'Use this to check a --layer-precision pattern without building.')
    parser.add_argument('--out', default=None, help='Write the full report.')
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.onnx and not args.engine:
        print('error: provide --onnx, --engine or both', file=sys.stderr)
        return 2
    report: Dict[str, Any] = {}
    summary: Dict[str, Any] = {}
    try:
        if args.onnx:
            report['graph'] = describe_graph(args.onnx, patterns=args.match)
            summary['graph'] = {
                key: report['graph'][key]
                for key in ('onnx', 'tensorrt_version', 'network_layers',
                            'layer_type_counts', 'pattern_matches')
            }
        if args.engine:
            report['engine'] = describe_engine(args.engine)
            summary['engine'] = {
                key: report['engine'][key]
                for key in ('engine', 'engine_bytes', 'tensorrt_version',
                            'engine_layers', 'datatype_mentions',
                            'datatype_is_available',
                            'layers_with_a_description', 'layer_type_counts',
                            'note')
            }
    except Exception as error:  # noqa: BLE001 - the CLI reports, never traces
        print(f'error: {type(error).__name__}: {error}', file=sys.stderr)
        return 1
    if args.out:
        target = Path(args.out).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
        summary['out'] = str(target)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    sys.exit(main())
