"""The layer report that turns a precision pattern into a checked fact.

The inspector's text parsing runs without TensorRT; the parse-and-count path
needs the real stack and is gated.
"""

import importlib.util
import json

import pytest
import torch

from tools import trt_layer_report

HAS_TENSORRT = importlib.util.find_spec('tensorrt') is not None
NEEDS_TENSORRT = pytest.mark.skipif(
    not HAS_TENSORRT, reason='Requires TensorRT.')

#: A DETAILED inspector layer description, in TensorRT's own shape: the
#: datatype marker lives inside the input and output tensor entries, and
#: LayerType names the implementation TensorRT selected.
_DETAILED_LAYER = json.dumps({
    'Name':
    '/model/blocks/blocks.0/norm1/LayerNormalization',
    'LayerType':
    'CaskNormalization',
    'Inputs': [{
        'Name': 'x',
        'Location': 'Device',
        'Dimensions': [1, 257, 1280],
        'Format/Datatype': 'Half',
    }],
    'Outputs': [{
        'Name': 'y',
        'Location': 'Device',
        'Dimensions': [1, 257, 1280],
        'Format/Datatype': 'Float',
    }],
    'TacticName':
    '...',
})

#: The same layer as reported by an engine built without DETAILED verbosity:
#: TensorRT returns the layer name and nothing else.
_NAMES_ONLY_LAYER = json.dumps(
    '/model/blocks/blocks.0/norm1/LayerNormalization')


def test_layer_information_reads_a_detailed_description():
    entry = trt_layer_report._layer_information(_DETAILED_LAYER)
    assert entry['name'] == '/model/blocks/blocks.0/norm1/LayerNormalization'
    assert entry['type'] == 'CaskNormalization'
    assert entry['input_datatypes'] == ['Half']
    assert entry['output_datatypes'] == ['Float']
    # A mixed layer is exactly what a precision override produces, so the
    # report keeps both sides.
    assert entry['datatypes'] == ['Float', 'Half']


def test_layer_information_reports_no_datatype_without_detailed_verbosity():
    entry = trt_layer_report._layer_information(_NAMES_ONLY_LAYER)
    assert entry['name'] == '/model/blocks/blocks.0/norm1/LayerNormalization'
    assert entry['type'] == '' and entry['datatypes'] == []
    # A name that is not JSON at all is still reported as a name.
    bare = trt_layer_report._layer_information('/model/a/LayerNormalization')
    assert bare['name'] == '/model/a/LayerNormalization'
    assert bare['datatypes'] == []


def test_layer_information_falls_back_to_text_parsing():
    entry = trt_layer_report._layer_information(
        'Name: /model/a/LayerNormalization, LayerType: Normalization, '
        '"Format/Datatype": "Half"')
    assert entry['datatypes'] == ['Half']


def test_matches_uses_glob_wildcards_like_trtexec():
    assert trt_layer_report._matches('/model/a/LayerNormalization',
                                     '*/LayerNormalization')
    assert trt_layer_report._matches('/model/a/LayerNormalization',
                                     '*/LayerNorm*')
    assert not trt_layer_report._matches('/model/a/LayerNormalization',
                                         'LayerNormalization')
    assert not trt_layer_report._matches('/model/a/Softmax',
                                         '*/LayerNormalization')


def test_the_cli_needs_something_to_look_at(capsys):
    assert trt_layer_report.main([]) == 2
    assert 'provide --onnx' in capsys.readouterr().err
    assert trt_layer_report.main(['--onnx', '/absent.onnx']) == 1
    assert 'FileNotFoundError' in capsys.readouterr().err


def test_parsing_a_graph_keeps_the_objects_the_network_points_into(
        monkeypatch, tmp_path):
    """The network outlives the builder, parser and logger that made it.

    TensorRT's network keeps references into those objects, so the parse
    helper has to hand them back with it instead of letting them be collected
    while the caller still reads the network.
    """
    import sys
    import types
    built = {}

    class FakeLayer:

        def __init__(self):
            self.name = '/model/a/Conv'
            self.type = 'CONVOLUTION'
            self.num_outputs = 1

        def get_output(self, index):
            return types.SimpleNamespace(dtype='FLOAT')

    class FakeNetwork:

        num_layers = 1

        def get_layer(self, index):
            return FakeLayer()

    class FakeParser:

        num_errors = 0

        def __init__(self, network, logger):
            built['parser'] = self
            built['network'] = network
            built['parse_logger'] = logger

        def parse_from_file(self, path):
            built['parsed_path'] = path
            return True

    class FakeBuilder:

        def __init__(self, logger):
            built['builder'] = self
            built['builder_logger'] = logger

        def create_network(self, flags=0):
            built['network_flags'] = flags
            return FakeNetwork()

    class FakeLogger:
        ERROR = 3

        def __init__(self, severity=3):
            built['logger'] = self
            self.severity = severity

    module = types.ModuleType('tensorrt')
    module.__version__ = '10.7.0'
    module.Logger = FakeLogger
    module.Builder = FakeBuilder
    module.OnnxParser = FakeParser
    module.NetworkDefinitionCreationFlag = types.SimpleNamespace(
        EXPLICIT_BATCH=1)
    monkeypatch.setitem(sys.modules, 'tensorrt', module)
    path = tmp_path / 'tiny.onnx'
    path.write_bytes(b'fake-onnx')

    session = trt_layer_report._parse_onnx_network(module, path)
    assert session.network is built['network']
    assert session.parser is built['parser']
    assert session.builder is built['builder']
    # One logger, shared by the builder and the parser, and returned with the
    # network so it cannot be collected first.
    assert session.logger is built['logger']
    assert built['builder_logger'] is session.logger
    assert built['parse_logger'] is session.logger
    # And it is really the parsed network the graph report reads.
    report = trt_layer_report.describe_graph(path)
    assert report['network_layers'] == 1
    assert report['layers'][0]['type'] == 'CONVOLUTION'
    assert report['layers'][0]['name'] == '/model/a/Conv'


@NEEDS_TENSORRT
def test_describe_graph_counts_pattern_matches(tmp_path):
    """The counting path is what makes a pattern safe to use."""
    pytest.importorskip('onnx')
    import torch.nn as nn
    model = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1)).eval()
    onnx_path = tmp_path / 'tiny.onnx'
    torch.onnx.export(
        model, (torch.rand(1, 3, 8, 8), ),
        str(onnx_path),
        input_names=['pixel_values'],
        output_names=['tokens'],
        external_data=False,
        dynamo=False)
    report = trt_layer_report.describe_graph(
        onnx_path, patterns=['*', 'Conv*', '/nothing/here'])
    assert report['network_layers'] == len(report['layers'])
    assert report['network_layers'] >= 1
    counts = {
        entry['pattern']: entry['matched_layers']
        for entry in report['pattern_matches']
    }
    # '*' matches every layer, an unmatched pattern is reported as zero rather
    # than raising, and a pattern that matches is counted.
    assert counts['*'] == report['network_layers']
    assert counts['/nothing/here'] == 0
    assert counts['Conv*'] + counts['/nothing/here'] <= report[
        'network_layers']
    assert sum(
        report['layer_type_counts'].values()) == report['network_layers']
    assert report['onnx_sha256']
    with pytest.raises(FileNotFoundError, match='ONNX model not found'):
        trt_layer_report.describe_graph(tmp_path / 'absent.onnx')


@NEEDS_TENSORRT
def test_describe_engine_reports_the_saved_engine(tmp_path):
    engine_path = (
        trt_layer_report.REPO_ROOT /
        'checkpoints/Pretrained_models/C-RADIOv4-H/exported/256x256/'
        'c_radio_v4_h_256.engine')
    if not engine_path.is_file():
        pytest.skip('The 256x256 C-RADIO engine is not present.')
    report = trt_layer_report.describe_engine(engine_path)
    assert report['engine_layers'] > 0
    assert report['engine_bytes'] == engine_path.stat().st_size
    # The shipped engine keeps names only, and the tool says so rather than
    # inventing a datatype.
    assert report['datatype_is_available'] is False
    assert report['datatype_mentions'] == {}
    assert report['layers_with_a_description'] == 0
    assert all(layer['datatypes'] == [] for layer in report['layers'])
    assert all(layer['name'] for layer in report['layers'])
    assert report['note']
    with pytest.raises(FileNotFoundError, match='Engine not found'):
        trt_layer_report.describe_engine(tmp_path / 'absent.engine')
