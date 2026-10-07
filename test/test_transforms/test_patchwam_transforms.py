# Copyright 2026 Limx Dynamics
"""CPU units for the PatchWAM image/packing/augmentation transforms."""

import pytest
import torch

from fluxvla.models.third_party_models.patchwam.data import (AppearanceRandomizer,
                                                             ClipAugment,
                                                             compose_cameras)
from fluxvla.transforms.transform_patchwam import (PatchWAMCleanVL,
                                                   PatchWAMComposeWorld,
                                                   PatchWAMEndpoints,
                                                   PatchWAMHistoryBuffer,
                                                   PatchWAMResizeCameras,
                                                   PatchWAMSelectHistory,
                                                   PatchWAMSelectWindow,
                                                   PatchWAMWorldAugment,
                                                   PatchWAMWorldHistory)


def world_frames(views=2, frames=17, height=6, width=8):
    return torch.rand(views, frames, 3, height, width)


def test_endpoints_select_current_and_future_rows():
    data = PatchWAMEndpoints(horizon=16)(
        dict(world_frames=world_frames(), patchwam_future_valid=True))
    endpoints = data['world_endpoints']
    assert endpoints.shape == (2, 2, 3, 6, 8)
    torch.testing.assert_close(endpoints[:, 1], data['world_frames'][:, 16])
    torch.testing.assert_close(endpoints[:, 0], data['world_frames'][:, 0])
    assert data['frame_masks'].tolist() == [True, True]
    assert data['img_masks'].all()

    invalid = PatchWAMEndpoints(horizon=16)(
        dict(world_frames=world_frames(), patchwam_future_valid=False))
    assert invalid['frame_masks'].tolist() == [True, False]


def test_endpoints_reject_a_short_window():
    try:
        PatchWAMEndpoints(horizon=16)(dict(world_frames=world_frames(frames=8)))
    except ValueError as error:
        assert 'endpoint' in str(error)
    else:
        raise AssertionError('short windows must be rejected')


def test_clean_vl_images_survive_world_augmentation():
    frames = world_frames()
    history = torch.rand(2, 3, 6, 8)
    data = dict(world_frames=frames, history_frames=history,
                patchwam_history_valid=torch.tensor([True, False]),
                patchwam_future_valid=True, patchwam_augment=True)
    data = PatchWAMEndpoints(horizon=16)(data)
    data = PatchWAMCleanVL(image_size=6)(data)
    clean_current = data['vl_current'].clone()
    clean_history = data['vl_history'].clone()
    data = PatchWAMWorldAugment(
        config=dict(p=1.0, augment_types=('both',),
                    color_jitter=dict(brightness=0.4),
                    gaussian_noise=dict(std=0.05)))(data)
    assert not torch.allclose(data['world_endpoints'], torch.stack(
        [frames[:, 0], frames[:, 16]], 1))
    torch.testing.assert_close(data['vl_current'], clean_current)
    torch.testing.assert_close(data['vl_history'], clean_history)
    assert not data['vl_history'][1].any()


def test_augmentation_is_disabled_outside_training():
    frames = world_frames()
    data = PatchWAMEndpoints(horizon=16)(
        dict(world_frames=frames, patchwam_future_valid=True,
             patchwam_augment=False))
    before = data['world_endpoints'].clone()
    data = PatchWAMWorldAugment(
        config=dict(p=1.0, color_jitter=dict(brightness=0.5)))(data)
    torch.testing.assert_close(data['world_endpoints'], before)


def test_clip_augmentation_matches_the_source_for_a_fixed_seed():
    config = dict(p=1.0, augment_types=('both',),
                  color_jitter=dict(brightness=0.3, contrast=0.2),
                  gamma=dict(range=[0.9, 1.1]),
                  exposure=dict(range=[-0.1, 0.1]),
                  gaussian_noise=dict(std=0.02),
                  random_resized_crop=dict(scale=[0.95, 1.0],
                                           ratio='preserve'),
                  rotate=dict(degrees=4.0))
    frames = world_frames(views=2, frames=17)
    history = torch.rand(2, 3, 6, 8)
    clips = [frames[0, (0, 16)], frames[1, (0, 16)],
             history[0].unsqueeze(0), history[1].unsqueeze(0)]
    torch.manual_seed(11)
    expected = ClipAugment(**config)(list(clips))
    data = dict(world_frames=frames, history_frames=history,
                patchwam_history_valid=torch.tensor([True, True]),
                patchwam_future_valid=True, patchwam_augment=True)
    data = PatchWAMEndpoints(horizon=16)(data)
    torch.manual_seed(11)
    data = PatchWAMWorldAugment(config=config)(data)
    torch.testing.assert_close(data['world_endpoints'][0], expected[0])
    torch.testing.assert_close(data['world_endpoints'][1], expected[1])
    torch.testing.assert_close(data['history_frames'][0], expected[2][0])
    torch.testing.assert_close(data['history_frames'][1], expected[3][0])


def test_appearance_augmentation_matches_the_source_for_a_fixed_seed():
    config = dict(p=1.0, photometric=dict(brightness=0.3),
                  style=dict(enabled=False), fourier=dict(enabled=False))
    frames = world_frames(views=1, frames=17)
    clips = [frames[0, (0, 16)]]
    torch.manual_seed(3)
    expected = AppearanceRandomizer(**config)(list(clips))
    data = dict(world_frames=frames, patchwam_future_valid=True,
                patchwam_augment=True)
    data = PatchWAMEndpoints(horizon=16)(data)
    torch.manual_seed(3)
    data = PatchWAMWorldAugment(config=config)(data)
    torch.testing.assert_close(data['world_endpoints'][0], expected[0])


def test_camera_geometry_matches_source_for_libero_and_robotwin():
    libero = [torch.rand(2, 3, 32, 48), torch.rand(2, 3, 32, 48)]
    data = PatchWAMEndpoints(horizon=16)(
        dict(world_frames=world_frames(views=2, height=32, width=48),
             patchwam_future_valid=True))
    data = PatchWAMComposeWorld(layout='horizontal',
                                video_size=(224, 448))(data)
    expected = compose_cameras(libero, 'horizontal', (224, 448),
                               'compact_288x256')
    assert data['video'].shape == expected.shape
    assert data['video'].shape == (3, 2, 224, 448)

    robotwin = [torch.rand(2, 3, 48, 64) for _ in range(3)]
    data = PatchWAMEndpoints(horizon=16)(
        dict(world_frames=world_frames(views=3, height=48, width=64),
             patchwam_future_valid=True))
    data = PatchWAMComposeWorld(layout='robotwin', video_size=(224, 448),
                                robotwin_layout='compact_288x256',
                                separate_views=True)(data)
    assert data['video'].shape == (3, 2, 224, 448)
    assert data['camera_video'].shape == (3, 3, 2, 48, 64)


def test_resize_cameras_sets_geometry_before_augmentation():
    world = torch.rand(3, 17, 3, 20, 24)
    history = torch.rand(5, 3, 20, 24)
    data = PatchWAMResizeCameras(camera_size=[16, 16])(
        dict(world_frames=world, history_frames=history))
    assert data['world_frames'].shape == (3, 17, 3, 16, 16)
    assert data['history_frames'].shape == (5, 3, 16, 16)
    data = PatchWAMEndpoints(horizon=16)(data)
    data = PatchWAMComposeWorld(layout='robotwin', video_size=[32, 32],
                                separate_views=True)(data)
    assert data['camera_video'].shape == (3, 3, 2, 16, 16)
    history = PatchWAMWorldHistory()(
        dict(world_endpoints=data['world_endpoints'],
             history_frames=torch.rand(5, 3, 16, 16),
             patchwam_history_valid=torch.ones(5, dtype=torch.bool)))
    assert history['history_video'].shape == (5, 3, 16, 16)


def test_separate_view_composition_matches_source_for_nonuniform_pixels():
    world = torch.rand(3, 17, 3, 24, 32)
    data = PatchWAMResizeCameras(camera_size=[16, 16])(
        dict(world_frames=world))
    data = PatchWAMEndpoints(horizon=16)(data)
    separate = PatchWAMComposeWorld(layout='robotwin', video_size=[32, 32],
                                    robotwin_layout='compact_288x256',
                                    separate_views=True)(dict(data))
    expected = compose_cameras(list(data['world_endpoints']), 'robotwin',
                               (32, 32), 'compact_288x256')
    torch.testing.assert_close(separate['video'], expected)


def test_world_history_masks_invalid_slots():
    frames = world_frames()
    data = PatchWAMEndpoints(horizon=16)(
        dict(world_frames=frames, history_frames=torch.rand(3, 3, 6, 8),
             patchwam_history_valid=torch.tensor([True, False, True]),
             patchwam_future_valid=True))
    data = PatchWAMWorldHistory()(data)
    assert data['history_video'].shape == (3, 3, 6, 8)
    assert not data['history_video'][1].any()
    assert data['history_video'][0].abs().max() <= 1


def test_row_selection_transforms_use_absolute_rows():
    decoded = torch.arange(20 * 3 * 4 * 5, dtype=torch.float32).reshape(
        20, 3, 4, 5)
    data = PatchWAMSelectWindow(horizon=16)(
        dict(decoded_frames=decoded, patchwam_row=2))
    assert data['world_frames'].shape == (1, 17, 3, 4, 5)
    torch.testing.assert_close(data['world_frames'][0, 0], decoded[2])
    torch.testing.assert_close(data['world_frames'][0, 16], decoded[18])

    rows = torch.tensor([3, 1])
    data = PatchWAMSelectHistory()(
        dict(decoded_frames=decoded, patchwam_history_rows=rows,
             patchwam_history_valid=torch.tensor([True, False])))
    assert data['history_frames'].shape == (2, 3, 4, 5)
    torch.testing.assert_close(data['history_frames'][0], decoded[3])
    assert not data['history_frames'][1].any()


def test_online_history_buffer_is_causal_and_resets_per_episode():
    buffer = PatchWAMHistoryBuffer(slots=2, interval_s=1.0,
                                   input_range='minus_one_one')
    frames = [torch.full((3, 4, 5), float(i) - 1.0) for i in range(3)]
    for step, frame in enumerate(frames):
        data = buffer(dict(image=frame, patchwam_episode_index=7,
                           patchwam_timestamp=float(step)))
    assert data['history_valid'].tolist() == [True, True]
    torch.testing.assert_close(data['history_video'][0], frames[1])
    torch.testing.assert_close(data['history_video'][1], frames[0])

    # A new episode clears history; only the newest frame is available.
    data = buffer(dict(image=frames[0], patchwam_episode_index=8,
                       patchwam_timestamp=0.0))
    assert not data['history_valid'].any()


def test_online_history_buffer_requires_signals():
    buffer = PatchWAMHistoryBuffer(slots=2, interval_s=1.0)
    try:
        buffer(dict(image=torch.zeros(3, 4, 5)))
    except ValueError as error:
        assert 'patchwam_episode_index' in str(error)
    else:
        raise AssertionError('missing cadence signals must be rejected')


def test_vl_history_uses_separate_rows_and_mask():
    decoded = torch.arange(20 * 3 * 4 * 5, dtype=torch.float32).reshape(
        20, 3, 4, 5)
    data = PatchWAMSelectHistory(
        rows_key='patchwam_vl_history_rows',
        valid_key='patchwam_vl_history_valid',
        out_key='vl_history_frames')(
            dict(decoded_frames=decoded,
                 patchwam_vl_history_rows=torch.tensor([2, 0]),
                 patchwam_vl_history_valid=torch.tensor([True, False])))
    assert data['vl_history_frames'].shape == (2, 3, 4, 5)
    torch.testing.assert_close(data['vl_history_frames'][0], decoded[2])
    assert not data['vl_history_frames'][1].any()
    data = PatchWAMCleanVL(image_size=4, history_key='vl_history_frames',
                           valid_key='patchwam_vl_history_valid')(
                               dict(world_frames=world_frames(height=4,
                                                              width=5),
                                    **data))
    assert data['vl_history_valid'].tolist() == [True, False]
    assert data['vl_history'].shape == (2, 3, 4, 4)


def test_online_history_buffer_accepts_all_documented_layouts():
    video = (torch.arange(5 * 3 * 4 * 6, dtype=torch.float32).reshape(
        5, 3, 4, 6) % 2) * 2 - 1  # signed values in [-1, 1]

    def drive(buffer, payload):
        payload = dict(payload)
        for step in range(3):
            payload['patchwam_episode_index'] = 1
            payload['patchwam_timestamp'] = float(step)
            out = buffer(payload)
        return out

    chw = drive(PatchWAMHistoryBuffer(slots=2, interval_s=1.0,
                                      input_range='minus_one_one'),
                dict(image=video[0]))
    torch.testing.assert_close(chw['history_video'][0], video[0])

    tchw = drive(PatchWAMHistoryBuffer(slots=2, interval_s=1.0,
                                       image_key='frames',
                                       frame_layout='t_c_h_w',
                                       input_range='minus_one_one'),
                 dict(frames=video))
    torch.testing.assert_close(tchw['history_video'][0], video[0])

    vtchw = drive(PatchWAMHistoryBuffer(slots=2, interval_s=1.0,
                                        image_key='frames',
                                        frame_layout='v_t_c_h_w',
                                        input_range='minus_one_one'),
                  dict(frames=video.unsqueeze(0)))
    torch.testing.assert_close(vtchw['history_video'][0], video[0])

    vchw = drive(PatchWAMHistoryBuffer(slots=2, interval_s=1.0,
                                       image_key='frames', frame_layout='v_c_h_w',
                                       input_range='minus_one_one'),
                 dict(frames=video[0].unsqueeze(0)))
    torch.testing.assert_close(vchw['history_video'][0], video[0])


def test_online_history_buffer_emits_clean_vl_history():
    buffer = PatchWAMHistoryBuffer(slots=2, interval_s=1.0, image_key='frames',
                                    vl_image_size=4)
    for step in range(3):
        out = buffer(dict(frames=torch.full((1, 1, 3, 8, 8), float(step)),
                          patchwam_episode_index=1,
                          patchwam_timestamp=float(step)))
    assert out['vl_history'].shape == (2, 3, 4, 4)
    assert out['vl_history_valid'].tolist() == [True, True]


def test_online_invalid_history_keeps_vl_mask_zeros_after_resize():
    buffer = PatchWAMHistoryBuffer(slots=3, interval_s=1.0, image_key='frames',
                                   vl_image_size=4, world_size=(4, 4),
                                   input_range='minus_one_one')
    for step in range(3):
        out = buffer(dict(frames=torch.full((1, 1, 3, 8, 8), float(step) - 1),
                          patchwam_episode_index=1,
                          patchwam_timestamp=float(step)))
    valid = out['vl_history_valid']
    assert valid.tolist() == [True, True, False]
    # Invalid slots stay exactly zero after the VL resize.
    assert not out['vl_history'][~valid].any()
    assert out['vl_history'][valid].abs().max() <= 1.001


def test_online_history_buffer_range_and_resize_contract():
    frame = torch.rand(3, 8, 8)  # [0, 1]
    buffer = PatchWAMHistoryBuffer(slots=2, interval_s=1.0,
                                   frame_layout='c_h_w', world_size=(4, 4),
                                   input_range='unit')
    for step in range(3):
        out = buffer(dict(image=frame, patchwam_episode_index=1,
                          patchwam_timestamp=float(step)))
    assert out['history_video'].shape == (2, 3, 4, 4)
    # [0, 1] input is stored in the training [-1, 1] range.
    assert out['history_video'].min() >= -1.001
    assert out['history_video'].min() <= 0.0

    uint8 = (torch.rand(3, 8, 8) * 255).to(torch.uint8)
    buffer = PatchWAMHistoryBuffer(slots=2, interval_s=1.0,
                                   frame_layout='c_h_w')
    for step in range(3):
        out = buffer(dict(image=uint8, patchwam_episode_index=1,
                          patchwam_timestamp=float(step)))
    assert out['history_video'].abs().max() <= 1.001


def test_online_history_buffer_rejects_ambiguous_4d_layout():
    buffer = PatchWAMHistoryBuffer(slots=2, interval_s=1.0,
                                   image_key='frames')
    with pytest.raises(ValueError, match='Ambiguous 4-D frame layout'):
        buffer(dict(frames=torch.zeros(4, 3, 8, 8),
                    patchwam_episode_index=1, patchwam_timestamp=0.0))


def test_online_history_buffer_reset_flag_clears_history():
    buffer = PatchWAMHistoryBuffer(slots=2, interval_s=1.0)
    for step in range(3):
        out = buffer(dict(image=torch.zeros(3, 4, 5),
                          patchwam_episode_index=1,
                          patchwam_timestamp=float(step)))
    assert out['history_valid'].any()
    out = buffer(dict(image=torch.zeros(3, 4, 5), patchwam_episode_index=1,
                      patchwam_timestamp=3.0, patchwam_reset=True))
    assert not out['history_valid'].any()
