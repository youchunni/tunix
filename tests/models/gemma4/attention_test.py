# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for Gemma 4 Attention module and Pallas Splash kernels."""

from __future__ import annotations

import dataclasses
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask as mask_lib
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
import numpy as np
from tunix.models.gemma4 import attention as attention_lib
from tunix.models.gemma4 import config as config_lib
from tunix.models.gemma4 import model as model_lib


class FlashAttentionMaskTest(parameterized.TestCase):
  """Mask correctness unit tests (pure numpy — no model needed)."""

  def test_local_mask_matches_manual(self):
    """Verify LocalMask with offset produces the correct sliding window mask."""
    chunk_len = 1024
    sw_size = 512
    cache_len = sw_size
    kv_len = cache_len + chunk_len
    prefix_len = cache_len

    # Splash mask with offset
    splash_mask = mask_lib.LocalMask(
        (chunk_len, kv_len),
        window_size=(sw_size - 1, 0),
        offset=prefix_len,
    )
    splash_array = splash_mask[np.s_[:, :]]

    # Reference mask for local sliding window with offset.
    position_offset = prefix_len
    valid_cache_len = prefix_len
    row_pos = np.arange(chunk_len) + position_offset
    col_pos_cache = np.arange(cache_len) + (position_offset - valid_cache_len)
    col_pos_suffix = np.arange(chunk_len) + position_offset
    col_pos = np.concatenate([col_pos_cache, col_pos_suffix])
    manual_mask = (col_pos[None, :] > (row_pos[:, None] - sw_size)) & (
        col_pos[None, :] <= row_pos[:, None]
    )

    np.testing.assert_array_equal(splash_array, manual_mask)

  def test_causal_mask_matches_manual(self):
    """Verify CausalMask with offset for GLOBAL chunked prefill."""
    chunk_len = 1024
    prefix_len = 2048
    kv_len = prefix_len + chunk_len

    splash_mask = mask_lib.CausalMask(
        (chunk_len, kv_len),
        offset=prefix_len,
    )
    splash_array = splash_mask[np.s_[:, :]]

    # Manual: q[i] can attend to kv[j] where i + offset >= j
    row = np.arange(chunk_len)[:, None] + prefix_len
    col = np.arange(kv_len)[None, :]
    manual_mask = row >= col

    np.testing.assert_array_equal(splash_array, manual_mask)

  @parameterized.parameters(
      # (chunk_len, sw_size) — various sizes to test edge cases
      (256, 128),
      (512, 256),
      (1024, 512),
      (2048, 1024),
  )
  def test_local_mask_offset_parameterized(self, chunk_len, sw_size):
    """LocalMask with offset is correct for various chunk/window sizes."""
    cache_len = sw_size
    kv_len = cache_len + chunk_len

    splash_mask = mask_lib.LocalMask(
        (chunk_len, kv_len),
        window_size=(sw_size - 1, 0),
        offset=cache_len,
    )
    splash_array = splash_mask[np.s_[:, :]]

    # Each Q position q[i] at logical position (i + cache_len) should attend
    # to KV positions in [i + cache_len - (sw_size - 1), i + cache_len].
    for i in range(0, chunk_len, max(1, chunk_len // 8)):
      logical_q = i + cache_len
      expected_start = max(0, logical_q - (sw_size - 1))
      expected_end = logical_q
      # Verify True positions in row i
      true_cols = np.where(splash_array[i])[0]
      if len(true_cols) > 0:
        self.assertEqual(true_cols[0], expected_start)
        self.assertEqual(true_cols[-1], expected_end)
        self.assertLen(true_cols, expected_end - expected_start + 1)

  def test_local_mask_square_no_offset(self):
    """Square LocalMask (chunk 1) should produce standard sliding window."""
    seq_len = 512
    sw_size = 128

    splash_mask = mask_lib.LocalMask(
        (seq_len, seq_len),
        window_size=(sw_size - 1, 0),
        offset=0,
    )
    splash_array = splash_mask[np.s_[:, :]]

    # Manual: standard causal sliding window
    row = np.arange(seq_len)[:, None]
    col = np.arange(seq_len)[None, :]
    manual_mask = (col <= row) & (col > row - sw_size)

    np.testing.assert_array_equal(splash_array, manual_mask)

  def test_build_flash_mask_local_sliding_rectangular(self):
    config = model_lib.ModelConfig.gemma4_e2b()
    config.sliding_window_size = 512
    attn = attention_lib.Attention(
        config=config,
        attn_type=model_lib.AttentionType.LOCAL_SLIDING,
        rngs=nnx.Rngs(0),
    )
    q_len, kv_len, sw = 128, 512, 512
    offset = kv_len - q_len
    mask = attn._build_flash_mask(q_len=q_len, kv_len=kv_len, offset=offset)
    mask_array = mask[np.s_[:, :]]

    q_ids = np.arange(q_len) + offset
    kv_ids = np.arange(kv_len)
    expected = (kv_ids[None, :] > (q_ids[:, None] - sw)) & (
        kv_ids[None, :] <= q_ids[:, None]
    )
    np.testing.assert_array_equal(mask_array, expected)

  def test_eager_attention_local_sliding_rectangular_mask(self):
    """Verify eager attention local sliding window mask in rectangular prefill."""
    q_len, kv_len, sw = 128, 512, 256
    offset = kv_len - q_len
    all_ones = jnp.ones((1, q_len, kv_len), dtype=jnp.bool_)
    sliding_mask = jnp.triu(all_ones, offset - sw + 1) * jnp.tril(
        all_ones, offset + sw - 1
    )

    q_ids = np.arange(q_len) + offset
    kv_ids = np.arange(kv_len)
    expected_sliding = (kv_ids[None, :] > (q_ids[:, None] - sw)) & (
        kv_ids[None, :] < (q_ids[:, None] + sw)
    )
    np.testing.assert_array_equal(sliding_mask[0], expected_sliding)

    # Combined with causal mask:
    causal_mask = jnp.tril(all_ones, offset)
    expected_causal_sliding = expected_sliding & (
        kv_ids[None, :] <= q_ids[:, None]
    )
    np.testing.assert_array_equal(
        (sliding_mask * causal_mask)[0], expected_causal_sliding
    )

  @parameterized.named_parameters(
      dict(testcase_name='2d_mask', mask_3d=False),
      dict(testcase_name='3d_mask', mask_3d=True),
  )
  def test_eager_attention_local_sliding_rectangular_execution(self, mask_3d):
    """Verify _eager_attention with local sliding window on rectangular shapes."""
    config = model_lib.ModelConfig.gemma4_e2b()
    config.sliding_window_size = 4
    attn = attention_lib.Attention(
        config=config,
        attn_type=model_lib.AttentionType.LOCAL_SLIDING,
        rngs=nnx.Rngs(0),
    )
    b, q_len, kv_len, d = 2, 4, 16, config.head_dim
    h, kh = config.num_heads, config.num_kv_heads
    offset = kv_len - q_len

    q = jax.random.normal(jax.random.PRNGKey(0), (b, q_len, h, d))
    k = jax.random.normal(jax.random.PRNGKey(1), (b, kv_len, kh, d))
    v = jax.random.normal(jax.random.PRNGKey(2), (b, kv_len, kh, d))
    mask_shape = (b, q_len, kv_len) if mask_3d else (q_len, kv_len)
    attn_mask = jnp.ones(mask_shape, dtype=jnp.bool_)
    segment_pos = jnp.broadcast_to(
        jnp.arange(offset, kv_len, dtype=jnp.int32)[None, :], (b, q_len)
    )

    out = attn._eager_attention(
        query_proj=q,
        key_proj=k,
        value_proj=v,
        attn_mask=attn_mask,
        segment_pos=segment_pos,
        cache=None,
        kv_shared_cache=None,
        seq_len=q_len,
    )
    self.assertEqual(out.shape, (b, q_len, h, d))
    self.assertFalse(jnp.isnan(out).any())


class FlashAttentionBlockSizeTest(parameterized.TestCase):
  """Block-size divisibility parameterized test."""

  @parameterized.parameters(
      model_lib.ModelConfig.gemma4_e2b,
      model_lib.ModelConfig.gemma4_e4b,
      model_lib.ModelConfig.gemma4_31b,
      model_lib.ModelConfig.gemma4_26b_a4b,
  )
  def test_block_kv_divisibility_and_chunk_multipliers(self, config_factory):
    """block_kv must be 128-aligned and divide kv_len across chunk sizes."""
    config = config_factory()
    sw = config.sliding_window_size
    block_q = config.flash_attention_block_size
    block_kv = min(block_q, sw)

    self.assertEqual(
        block_kv % 128,
        0,
        f'block_kv={block_kv} not a multiple of 128 (NUM_LANES)',
    )

    for multiplier in (1, 2, 4):
      chunk_len = block_q * multiplier
      kv_len = sw + chunk_len
      self.assertEqual(
          chunk_len % block_q,
          0,
          f'chunk_len={chunk_len} not divisible by block_q={block_q}',
      )
      self.assertEqual(
          kv_len % block_kv,
          0,
          f'kv_len={kv_len} not divisible by block_kv={block_kv} '
          f'(multiplier={multiplier})',
      )


class AttentionTest(parameterized.TestCase):

  def test_attention_with_segment_ids_rectangular_routing(self):
    config = model_lib.ModelConfig.gemma4_e2b()
    config.use_flash_attention = True
    config.flash_attention_block_size = 16

    attn = attention_lib.Attention(
        config=config,
        attn_type=model_lib.AttentionType.LOCAL_SLIDING,
        rngs=nnx.Rngs(0),
    )

    b, t, h, d = 2, 32, config.num_heads, config.head_dim
    x = jnp.zeros((b, t, config.embed_dim))
    segment_pos = jnp.zeros((b, t), dtype=jnp.int32)
    attn_mask = jnp.ones((b, t, t), dtype=jnp.bool_)

    # Case 1: Square sequence, segment_ids is not None -> should use FLASH
    with mock.patch.object(
        attn, '_flash_attention_single'
    ) as mock_flash, mock.patch.object(
        attn, '_eager_attention'
    ) as mock_eager, mock.patch.object(
        attn, '_make_sharding_specs'
    ) as mock_sharding, mock.patch.object(
        attn, '_make_splash_kernel'
    ) as mock_kernel:

      mock_flash.return_value = (
          jnp.zeros((b, t, h, d)),
          jnp.zeros((b, t, config.num_kv_heads, d)),
          jnp.zeros((b, t, config.num_kv_heads, d)),
      )
      mock_eager.return_value = jnp.zeros((b, t, h, d))
      mock_sharding.return_value = (None,) * 4 + (1, 1) + (None,) * 3
      mock_kernel.return_value = (None, None)

      segment_ids = jnp.zeros((b, t), dtype=jnp.int32)

      attn.block(
          x,
          segment_pos,
          cache=None,
          attn_mask=attn_mask,
          segment_ids=segment_ids,
      )

      mock_flash.assert_called_once()
      mock_eager.assert_not_called()

    # Case 2: Rectangular sequence, segment_ids is not None -> should use EAGER
    kv_len = 64
    kv_shared_cache = {
        'k': jnp.zeros((b, kv_len, config.num_kv_heads, d)),
        'v': jnp.zeros((b, kv_len, config.num_kv_heads, d)),
    }
    attn_mask_rect = jnp.ones((b, t, kv_len), dtype=jnp.bool_)

    with mock.patch.object(
        attn, '_flash_attention_single'
    ) as mock_flash, mock.patch.object(attn, '_eager_attention') as mock_eager:

      mock_flash.return_value = (
          jnp.zeros((b, t, h, d)),
          jnp.zeros((b, kv_len, config.num_kv_heads, d)),
          jnp.zeros((b, kv_len, config.num_kv_heads, d)),
      )
      mock_eager.return_value = jnp.zeros((b, t, h, d))

      segment_ids = jnp.zeros((b, t), dtype=jnp.int32)

      attn.block(
          x,
          segment_pos,
          cache=None,
          attn_mask=attn_mask_rect,
          kv_shared_cache=kv_shared_cache,
          segment_ids=segment_ids,
      )

      mock_flash.assert_not_called()
      mock_eager.assert_called_once()

  def test_make_block_sizes(self):
    config = model_lib.ModelConfig.gemma4_e2b()
    config.flash_attention_block_size = 128
    config.sliding_window_size = 64
    self.assertGreater(
        config.flash_attention_block_size, config.sliding_window_size
    )

    global_attn = attention_lib.Attention(
        config=config,
        attn_type=model_lib.AttentionType.GLOBAL,
        rngs=nnx.Rngs(0),
    )
    local_attn = attention_lib.Attention(
        config=config,
        attn_type=model_lib.AttentionType.LOCAL_SLIDING,
        rngs=nnx.Rngs(0),
    )

    # GLOBAL rectangular uses the full block size (kills `if is_rectangular:`).
    self.assertEqual(
        global_attn._make_block_sizes(is_rectangular=True).block_kv,
        config.flash_attention_block_size,
    )

    # LOCAL_SLIDING square uses the full block size (kills `if self.attn_type == LOCAL_SLIDING:`).
    self.assertEqual(
        local_attn._make_block_sizes(is_rectangular=False).block_kv,
        config.flash_attention_block_size,
    )

    # LOCAL_SLIDING rectangular uses min(block_size, window_size).
    self.assertEqual(
        local_attn._make_block_sizes(is_rectangular=True).block_kv,
        min(config.flash_attention_block_size, config.sliding_window_size),
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='none_mesh',
          act_btnh=None,
          mesh_shape=None,
          expected_head_shards=1,
          expected_q_seq_shards=1,
      ),
      dict(
          testcase_name='axis_not_in_mesh_defaults_to_one',
          # act_btnh unpacks to (shd_b, shd_t, shd_n, shd_h). Use axis names for
          # shd_t ('seq_axis') and shd_n ('model_axis') that are non-None but
          # NOT present in the mesh.
          act_btnh=P('fsdp', 'seq_axis', 'model_axis', None),
          # Mesh axes are ('fsdp', 'x'); 'seq_axis' and 'model_axis' are absent.
          mesh_shape={'fsdp': 1, 'x': 1},
          # shd_n ('model_axis') and shd_t ('seq_axis') are not in the mesh, so
          # both must fall back to 1. This kills the mutants that drop the
          # `and shd_n in mesh.shape` / `and shd_t in mesh.shape` guards.
          expected_head_shards=1,
          expected_q_seq_shards=1,
      ),
      dict(
          testcase_name='sharded_mesh',
          act_btnh=P('fsdp', 'seq_axis', 'model_axis', None),
          mesh_shape={'fsdp': 1, 'seq_axis': 4, 'model_axis': 2},
          expected_head_shards=2,
          expected_q_seq_shards=4,
      ),
  )
  def test_make_sharding_specs(
      self,
      act_btnh,
      mesh_shape,
      expected_head_shards,
      expected_q_seq_shards,
  ):
    config = model_lib.ModelConfig.gemma4_e2b()
    if act_btnh is not None:
      # ShardingConfig is frozen, so replace it.
      config.shd_config = dataclasses.replace(
          config.shd_config,
          act_btnh=act_btnh,
      )
    attn = attention_lib.Attention(
        config=config,
        attn_type=model_lib.AttentionType.GLOBAL,
        rngs=nnx.Rngs(0),
    )
    b = 1
    kh = config.num_kv_heads
    mesh = mock.MagicMock(shape=mesh_shape) if mesh_shape is not None else None
    if mesh is not None and 'fsdp' in mesh.shape:
      self.assertEqual(b % mesh.shape['fsdp'], 0)

    specs = attn._make_sharding_specs(b, kh, mesh)
    # Return tuple index 4 is head_shards, index 5 is q_seq_shards.
    head_shards = specs[4]
    q_seq_shards = specs[5]

    self.assertEqual(head_shards, expected_head_shards)
    self.assertEqual(q_seq_shards, expected_q_seq_shards)

  def test_attention_flash_rectangular_offset(self):
    config = model_lib.ModelConfig.gemma4_e2b()
    config.use_flash_attention = True
    config.flash_attention_block_size = 16

    attn = attention_lib.Attention(
        config=config,
        attn_type=model_lib.AttentionType.LOCAL_SLIDING,
        rngs=nnx.Rngs(0),
    )

    b, q_len, kv_len = 2, 32, 64
    h, d = config.num_heads, config.head_dim
    x = jnp.zeros((b, q_len, config.embed_dim))
    segment_pos = jnp.zeros((b, q_len), dtype=jnp.int32)
    attn_mask = jnp.ones((b, q_len, kv_len), dtype=jnp.bool_)
    kv_shared_cache = {
        'k': jnp.zeros((b, kv_len, config.num_kv_heads, d)),
        'v': jnp.zeros((b, kv_len, config.num_kv_heads, d)),
    }

    with mock.patch.object(
        attn, '_build_flash_mask', wraps=attn._build_flash_mask
    ) as mock_mask, mock.patch.object(
        attn, '_flash_attention_single'
    ) as mock_flash, mock.patch.object(
        attn, '_make_sharding_specs'
    ) as mock_sharding, mock.patch.object(
        attn, '_make_splash_kernel'
    ) as mock_kernel:

      mock_flash.return_value = (
          jnp.zeros((b, q_len, h, d)),
          jnp.zeros((b, kv_len, config.num_kv_heads, d)),
          jnp.zeros((b, kv_len, config.num_kv_heads, d)),
      )
      mock_sharding.return_value = (None,) * 4 + (1, 1) + (None,) * 3
      mock_kernel.return_value = (None, None)

      attn.block(
          x,
          segment_pos,
          cache=None,
          attn_mask=attn_mask,
          kv_shared_cache=kv_shared_cache,
          segment_ids=None,
      )

      # Verify flash attention is called and the exact positive offset is passed
      # (kv_len - q_len = 64 - 32 = 32), killing mutants that negate offset or
      # pass 0.
      mock_flash.assert_called_once()
      mock_mask.assert_called_once_with(q_len, kv_len, kv_len - q_len)

  @parameterized.named_parameters(
      dict(
          testcase_name='own_cache',
          # Case 1: Own cache is present (cache is not None).
          # end_index = 4 -> position 4 is valid and attended to.
          end_index=4,
          use_shared_cache=False,
      ),
      dict(
          testcase_name='shared_cache_adjusted_index',
          # Case 2: Shared cache is present (cache is None, kv_shared_cache is
          # not None). end_index = 5 -> adjusted to 4 by `end_idx - 1`.
          end_index=5,
          use_shared_cache=True,
      ),
  )
  def test_eager_attention_decoding_sliding_window_cache_indexing(
      self, end_index, use_shared_cache
  ):
    config = model_lib.ModelConfig.gemma4_e2b()
    config.sliding_window_size = 8
    config.use_sliding_window_kv_cache = True
    attn = attention_lib.Attention(
        config=config,
        attn_type=model_lib.AttentionType.LOCAL_SLIDING,
        rngs=nnx.Rngs(0),
    )
    b, q_len, cache_len, d = 1, 1, 8, config.head_dim
    h, kh = config.num_heads, config.num_kv_heads

    q = jnp.ones((b, q_len, h, d))
    k = jnp.zeros((b, cache_len, kh, d))
    k = k.at[:, 4, :, :].set(10.0)
    k = k.at[:, 6, :, :].set(20.0)
    v = jnp.ones((b, cache_len, kh, d))
    v = v.at[:, 4, :, :].set(5.0)
    v = v.at[:, 6, :, :].set(100.0)

    attn_mask = jnp.ones((b, q_len, cache_len), dtype=jnp.bool_)
    segment_pos = jnp.array([[4]], dtype=jnp.int32)

    cache_dict = {'end_index': jnp.array([end_index])}
    cache = None if use_shared_cache else cache_dict
    kv_shared_cache = cache_dict if use_shared_cache else None

    out = attn._eager_attention(
        query_proj=q,
        key_proj=k,
        value_proj=v,
        attn_mask=attn_mask,
        segment_pos=segment_pos,
        cache=cache,
        kv_shared_cache=kv_shared_cache,
        seq_len=1,
    )
    np.testing.assert_allclose(out, jnp.full_like(out, 5.0), atol=1e-2)

  def test_sliding_window_kv_cache_prefill_over_cache_len(self):
    config = model_lib.ModelConfig.gemma4_e2b()
    config.sliding_window_size = 8
    b, seq_len, cache_len = 1, 10, 8
    d = config.head_dim
    kh = config.num_kv_heads

    x = jax.random.normal(jax.random.PRNGKey(0), (b, seq_len, config.embed_dim))
    segment_pos = jnp.arange(seq_len, dtype=jnp.int32)[None, :]
    attn_mask = jnp.ones((b, seq_len, seq_len), dtype=jnp.bool_)
    cache = {
        'k': jnp.zeros((b, cache_len, kh, d)),
        'v': jnp.zeros((b, cache_len, kh, d)),
        'end_index': jnp.zeros((b,), dtype=jnp.int32),
    }

    # 1. When use_sliding_window_kv_cache=True, circular update succeeds.
    config.use_sliding_window_kv_cache = True
    attn_sliding = attention_lib.Attention(
        config=config,
        attn_type=model_lib.AttentionType.LOCAL_SLIDING,
        rngs=nnx.Rngs(0),
    )
    new_cache, _, (k_proj, v_proj, *_) = attn_sliding.block(
        x,
        segment_pos,
        cache=cache,
        attn_mask=attn_mask,
        force_eager=True,
    )

    self.assertIsNotNone(new_cache)
    self.assertEqual(new_cache['end_index'][0], seq_len)
    valid_indices = (seq_len - cache_len + jnp.arange(cache_len)) % cache_len
    np.testing.assert_allclose(
        new_cache['k'][:, valid_indices, ...], k_proj[:, -cache_len:, ...]
    )
    np.testing.assert_allclose(
        new_cache['v'][:, valid_indices, ...], v_proj[:, -cache_len:, ...]
    )

    # 2. When use_sliding_window_kv_cache=False, non-sliding prefill cannot
    # exceed cache_len (dynamic_update_slice raises). This kills the mutant
    # at line 419 that drops `self.config.use_sliding_window_kv_cache and`.
    config.use_sliding_window_kv_cache = False
    attn_standard = attention_lib.Attention(
        config=config,
        attn_type=model_lib.AttentionType.LOCAL_SLIDING,
        rngs=nnx.Rngs(0),
    )
    with self.assertRaises(TypeError):
      attn_standard.block(
          x,
          segment_pos,
          cache=cache,
          attn_mask=attn_mask,
          force_eager=True,
      )

  def test_eager_attention_mha_non_gqa(self):
    config = model_lib.ModelConfig.gemma4_e2b()
    config.num_heads = 4
    config.num_kv_heads = 4
    attn = attention_lib.Attention(
        config=config,
        attn_type=model_lib.AttentionType.GLOBAL,
        rngs=nnx.Rngs(0),
    )
    self.assertFalse(attn.use_gqa)
    b, q_len, kv_len, d = 2, 4, 8, config.head_dim
    h = config.num_heads

    q = jnp.ones((b, q_len, h, d))
    k = jnp.zeros((b, kv_len, h, d))
    k = k.at[:, 2, :, :].set(10.0)
    v = jnp.zeros((b, kv_len, h, d))
    v = v.at[:, 2, :, :].set(3.0)

    attn_mask = jnp.ones((b, q_len, kv_len), dtype=jnp.bool_)
    segment_pos = jnp.broadcast_to(
        jnp.arange(q_len, dtype=jnp.int32)[None, :], (b, q_len)
    )

    out = attn._eager_attention(
        query_proj=q,
        key_proj=k,
        value_proj=v,
        attn_mask=attn_mask,
        segment_pos=segment_pos,
        cache=None,
        kv_shared_cache=None,
        seq_len=q_len,
    )
    self.assertEqual(out.shape, (b, q_len, h, d))
    np.testing.assert_allclose(out, jnp.full_like(out, 3.0), atol=1e-2)

  def test_kv_cache_prefill_within_cache_len_and_decode_step(self):
    config = model_lib.ModelConfig.gemma4_e2b()
    config.use_sliding_window_kv_cache = False
    attn = attention_lib.Attention(
        config=config,
        attn_type=model_lib.AttentionType.GLOBAL,
        rngs=nnx.Rngs(0),
    )
    b, prefill_len, cache_len = 1, 4, 16
    d = attn.head_dim
    kh = attn.num_kv_heads

    # 1. Prefill step with seq_len <= cache_len
    x_prefill = jax.random.normal(
        jax.random.PRNGKey(0), (b, prefill_len, config.embed_dim)
    )
    pos_prefill = jnp.arange(prefill_len, dtype=jnp.int32)[None, :]
    mask_prefill = jnp.ones((b, prefill_len, prefill_len), dtype=jnp.bool_)
    cache = {
        'k': jnp.zeros((b, cache_len, kh, d)),
        'v': jnp.zeros((b, cache_len, kh, d)),
        'end_index': jnp.zeros((b,), dtype=jnp.int32),
    }

    cache, _, (k_prefill, v_prefill, *_) = attn.block(
        x_prefill,
        pos_prefill,
        cache=cache,
        attn_mask=mask_prefill,
        force_eager=True,
    )
    self.assertEqual(cache['end_index'][0], prefill_len)
    np.testing.assert_allclose(cache['k'][:, :prefill_len, ...], k_prefill)
    np.testing.assert_allclose(cache['v'][:, :prefill_len, ...], v_prefill)

    # 2. Decode step with seq_len == 1
    x_decode = jax.random.normal(
        jax.random.PRNGKey(1), (b, 1, config.embed_dim)
    )
    pos_decode = jnp.array([[prefill_len]], dtype=jnp.int32)
    mask_decode = jnp.ones((b, 1, cache_len), dtype=jnp.bool_)

    cache, _, (k_decode, v_decode, *_) = attn.block(
        x_decode,
        pos_decode,
        cache=cache,
        attn_mask=mask_decode,
        force_eager=True,
    )
    self.assertEqual(cache['end_index'][0], prefill_len + 1)
    np.testing.assert_allclose(cache['k'], k_decode)
    np.testing.assert_allclose(cache['v'], v_decode)

  def test_eager_attention_decoding_without_sliding_window_cache(self):
    config = model_lib.ModelConfig.gemma4_e2b()
    config.sliding_window_size = 4
    config.use_sliding_window_kv_cache = False
    attn = attention_lib.Attention(
        config=config,
        attn_type=model_lib.AttentionType.LOCAL_SLIDING,
        rngs=nnx.Rngs(0),
    )
    b, q_len, kv_len, d = 1, 1, 8, config.head_dim
    h, kh = config.num_heads, config.num_kv_heads

    q = jnp.ones((b, q_len, h, d))
    k = jnp.zeros((b, kv_len, kh, d))
    # Query at position 6 with window 4 -> valid window [3, 6].
    k = k.at[:, 5, :, :].set(10.0)
    k = k.at[:, 1, :, :].set(20.0)
    v = jnp.zeros((b, kv_len, kh, d))
    v = v.at[:, 5, :, :].set(4.0)
    v = v.at[:, 1, :, :].set(99.0)

    # Causal mask up to position 6.
    attn_mask = (
        jnp.zeros((b, q_len, kv_len), dtype=jnp.bool_).at[:, :, :7].set(True)
    )
    segment_pos = jnp.array([[6]], dtype=jnp.int32)

    out = attn._eager_attention(
        query_proj=q,
        key_proj=k,
        value_proj=v,
        attn_mask=attn_mask,
        segment_pos=segment_pos,
        cache=None,
        kv_shared_cache=None,
        seq_len=1,
    )
    self.assertEqual(out.shape, (b, q_len, h, d))
    np.testing.assert_allclose(out, jnp.full_like(out, 4.0), atol=1e-2)

  def test_eager_attention_decoding_missing_cache_raises(self):
    config = model_lib.ModelConfig.gemma4_e2b()
    config.sliding_window_size = 4
    config.use_sliding_window_kv_cache = True
    attn = attention_lib.Attention(
        config=config,
        attn_type=model_lib.AttentionType.LOCAL_SLIDING,
        rngs=nnx.Rngs(0),
    )
    b, q_len, kv_len, d = 1, 1, 8, config.head_dim
    h, kh = config.num_heads, config.num_kv_heads

    q = jnp.ones((b, q_len, h, d))
    k = jnp.ones((b, kv_len, kh, d))
    v = jnp.ones((b, kv_len, kh, d))
    attn_mask = jnp.ones((b, q_len, kv_len), dtype=jnp.bool_)
    segment_pos = jnp.array([[4]], dtype=jnp.int32)

    with self.assertRaisesRegex(
        ValueError, 'Cache or shared cache is required'
    ):
      attn._eager_attention(
          query_proj=q,
          key_proj=k,
          value_proj=v,
          attn_mask=attn_mask,
          segment_pos=segment_pos,
          cache=None,
          kv_shared_cache=None,
          seq_len=1,
      )

  def test_find_last_one_index(self):
    mask = jnp.array(
        [
            [[1, 1, 1, 0, 0]],
            [[1, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0]],
        ],
        dtype=jnp.int32,
    )
    last_indices = attention_lib.find_last_one_index(mask)
    np.testing.assert_array_equal(
        last_indices,
        np.array([2, 0, 0], dtype=np.int32),
    )

  def test_create_logical_sliding_window_mask_gapped_boundaries(self):
    """Verify create_logical_sliding_window_mask with gapped mask."""
    # Mask with 6 valid slots across a gap: [0, 1, 2, 3, 4, 14]
    cache_len = 16
    sw = 4
    mask_indices = [0, 1, 2, 3, 4, 14]
    attn_mask = jnp.zeros((1, 1, cache_len), dtype=jnp.int32)
    attn_mask = attn_mask.at[0, 0, mask_indices].set(1)

    result = attention_lib.create_logical_sliding_window_mask(
        attn_mask, sliding_window_size=sw
    )

    # Valid count = 6, logical_last = 5, threshold = logical_last - sw = 1.
    # Slot 1 has logical pos 1 (<= 1 threshold) -> False.
    # Slot 2 has logical pos 2 (> 1 threshold) -> True.
    # Slots 2, 3, 4, 14 are within the logical window of size 4.
    self.assertFalse(bool(result[0, 0, 1]))
    self.assertTrue(bool(result[0, 0, 2]))
    self.assertEqual(int(jnp.sum(result)), 4)
    expected = jnp.zeros((1, 1, cache_len), dtype=jnp.bool_)
    expected = expected.at[0, 0, [2, 3, 4, 14]].set(True)
    np.testing.assert_array_equal(result, expected)

  def test_create_logical_sliding_window_mask_contiguous_matches_physical(self):
    """Contiguous mask produces identical result to physical sliding window."""
    attn_mask = jnp.array([[[1, 1, 1, 1, 0, 0]]], dtype=jnp.int32)
    sw = 2
    logical_mask = attention_lib.create_logical_sliding_window_mask(
        attn_mask, sliding_window_size=sw
    )
    physical_mask = attention_lib.create_sliding_window_mask(
        attn_mask, sliding_window_size=sw
    )
    np.testing.assert_array_equal(logical_mask, physical_mask)

  def test_has_physical_gap_batched(self):
    """Verify _has_physical_gap identifies gapped vs contiguous masks in batch."""
    cache_len = 16
    attn_mask = jnp.zeros((7, 1, cache_len), dtype=jnp.int32)
    # row 0: contiguous [1, 1, 1, 1, 0, ...] -> False
    attn_mask = attn_mask.at[0, 0, [0, 1, 2, 3]].set(1)
    # row 1: multi-gap [1, 1, 0, 0, 1, 0, ...] -> True
    attn_mask = attn_mask.at[1, 0, [0, 1, 4]].set(1)
    # row 2: all zeros [0, 0, 0, ...] -> False
    # row 3: single token [0, 0, 1, 0, ...] -> False
    attn_mask = attn_mask.at[3, 0, [2]].set(1)
    # row 4: single-token gap [1, 0, 1, 0, ...] (count=2, span=3) -> True
    # Kills mutant that computes span = last - first without + 1 (2 < 2 -> False)
    attn_mask = attn_mask.at[4, 0, [0, 2]].set(1)
    # row 5: single-token gap with prefix [1, 1, 0, 1, 0, ...] (count=3, span=4) -> True
    attn_mask = attn_mask.at[5, 0, [0, 1, 3]].set(1)
    # row 6: offset contiguous [0, 0, 1, 1, 1, 0, ...] (count=3, span=3) -> False
    attn_mask = attn_mask.at[6, 0, [2, 3, 4]].set(1)

    result = attention_lib._has_physical_gap(attn_mask)
    self.assertEqual(result.shape, (7, 1, 1))
    expected = jnp.array([
        [[False]],
        [[True]],
        [[False]],
        [[False]],
        [[True]],
        [[True]],
        [[False]],
    ])
    np.testing.assert_array_equal(result, expected)

  def test_read_prefix_kv_local_sliding_ring_buffer_unrolling(self):
    """Verify unrolling of ring buffer in _read_prefix_kv for LOCAL_SLIDING."""
    config = model_lib.ModelConfig.gemma4_e2b()
    config.use_sliding_window_kv_cache = True
    config.sliding_window_size = 8
    attn = attention_lib.Attention(
        config=config,
        attn_type=model_lib.AttentionType.LOCAL_SLIDING,
        rngs=nnx.Rngs(0),
    )
    b, cache_len, seq_len = 1, 8, 4
    kh, d = config.num_kv_heads, config.head_dim
    # Cache slots 0..7 with recognizable distinct values (e.g. 0.0, 10.0, 20.0, ...)
    k_cached = jnp.broadcast_to(
        (jnp.arange(cache_len, dtype=jnp.float32) * 10.0)[None, :, None, None],
        (b, cache_len, kh, d),
    )
    v_cached = jnp.broadcast_to(
        (jnp.arange(cache_len, dtype=jnp.float32) * 10.0 + 1.0)[
            None, :, None, None
        ],
        (b, cache_len, kh, d),
    )
    cache = {
        'k': k_cached,
        'v': v_cached,
        'end_index': jnp.array([12]),
    }
    key_proj = jnp.full((b, seq_len, kh, d), 100.0)
    value_proj = jnp.full((b, seq_len, kh, d), 101.0)
    prior_end_index = jnp.array([12])

    res = attn._read_prefix_kv(
        cache,
        key_proj,
        value_proj,
        seq_len,
        is_chunked_prefill=True,
        prefix_length=8,
        prior_end_index=prior_end_index,
    )
    k_out, v_out, kv_valid_mask = res[0], res[1], res[2]

    # For cache_len=8, prior_end_index=12:
    # valid_cached = 8, read_start = (12 - 8) % 8 = 4.
    # Unrolled order of physical slots is [4, 5, 6, 7, 0, 1, 2, 3].
    expected_k_prefix = jnp.array(
        [40.0, 50.0, 60.0, 70.0, 0.0, 10.0, 20.0, 30.0]
    )
    expected_v_prefix = jnp.array(
        [41.0, 51.0, 61.0, 71.0, 1.0, 11.0, 21.0, 31.0]
    )

    self.assertEqual(k_out.shape, (b, cache_len + seq_len, kh, d))
    self.assertEqual(v_out.shape, (b, cache_len + seq_len, kh, d))
    np.testing.assert_allclose(k_out[0, :cache_len, 0, 0], expected_k_prefix)
    np.testing.assert_allclose(v_out[0, :cache_len, 0, 0], expected_v_prefix)
    np.testing.assert_allclose(
        k_out[0, cache_len:, 0, 0], jnp.full((seq_len,), 100.0)
    )
    np.testing.assert_allclose(
        v_out[0, cache_len:, 0, 0], jnp.full((seq_len,), 101.0)
    )
    self.assertIsNotNone(kv_valid_mask)
    np.testing.assert_array_equal(
        kv_valid_mask, jnp.ones((cache_len,), dtype=jnp.bool_)
    )

  def test_eager_attention_chunked_prefill_mask_construction(self):
    """Verify _eager_attention constructs chunked prefill mask for rectangular KV."""
    config = model_lib.ModelConfig.gemma4_e2b()
    config.sliding_window_size = 8
    attn = attention_lib.Attention(
        config=config,
        attn_type=model_lib.AttentionType.GLOBAL,
        rngs=nnx.Rngs(0),
    )
    b, q_len, kv_len = 2, 4, 12
    h, kh, d = config.num_heads, config.num_kv_heads, config.head_dim
    q = jnp.ones((b, q_len, h, d))
    k = jnp.zeros((b, kv_len, kh, d))
    v = jnp.zeros((b, kv_len, kh, d))

    # Active Probe at valid prefix slot 2 (< prior_end_index=6)
    k = k.at[:, 2, :, :].set(10.0)
    v = v.at[:, 2, :, :].set(5.0)

    # Trap Probe 1 at uninitialized prefix position 7 (> prior_end_index=6)
    k = k.at[:, 7, :, :].set(100.0)
    v = v.at[:, 7, :, :].set(999.0)

    # Trap Probe 2 at future suffix position 11 (for query 0)
    k = k.at[:, 11, :, :].set(100.0)
    v = v.at[:, 11, :, :].set(999.0)

    prefix_mask = jnp.ones((b, q_len, 8), dtype=jnp.bool_)
    suffix_causal = jnp.broadcast_to(
        jnp.tril(jnp.ones((q_len, q_len), dtype=jnp.bool_))[None, :, :],
        (b, q_len, q_len),
    )
    attn_mask = jnp.concatenate([prefix_mask, suffix_causal], axis=-1)
    segment_pos = jnp.broadcast_to(
        jnp.arange(8, 12, dtype=jnp.int32)[None, :], (b, q_len)
    )
    cache = {
        'k': jnp.zeros((b, 8, kh, d)),
        'v': jnp.zeros((b, 8, kh, d)),
        'end_index': jnp.array([6]),
    }
    out = attn._eager_attention(
        query_proj=q,
        key_proj=k,
        value_proj=v,
        attn_mask=attn_mask,
        segment_pos=segment_pos,
        cache=cache,
        kv_shared_cache=None,
        prior_end_index=jnp.array([6]),
        prefix_length=8,
        seq_len=q_len,
        is_chunked_prefill=True,
    )
    self.assertEqual(out.shape, (b, q_len, h, d))
    self.assertFalse(jnp.isnan(out).any())
    np.testing.assert_allclose(out[:, 0, :, :], 5.0, atol=1e-2)

  def test_flash_attention_single_with_segment_ids(self):
    """Verify _flash_attention_single execution path with segment_ids."""
    config = model_lib.ModelConfig.gemma4_e2b()
    config.use_flash_attention = True
    config.flash_attention_block_size = 16
    attn = attention_lib.Attention(
        config=config,
        attn_type=model_lib.AttentionType.GLOBAL,
        rngs=nnx.Rngs(0),
    )
    b, t = 2, 16
    x = jnp.zeros((b, t, config.embed_dim))
    segment_pos = jnp.zeros((b, t), dtype=jnp.int32)
    attn_mask = jnp.ones((b, t, t), dtype=jnp.bool_)
    segment_ids = jnp.zeros((b, t), dtype=jnp.int32)

    kernel_called = []

    def mock_kernel(q_in, k_in, v_in, segment_ids=None):
      kernel_called.append(segment_ids)
      self.assertIsNotNone(segment_ids)
      return jnp.zeros_like(q_in)

    devices = np.array(jax.devices()[:1]).reshape(1, 1)
    mesh = jax.sharding.Mesh(devices, ('fsdp', 'tp'))

    with mesh, mock.patch.object(
        attn, '_make_splash_kernel', return_value=(mock_kernel, None)
    ):
      new_cache, out, (k_proj, v_proj, *_) = attn.block(
          x,
          segment_pos,
          cache=None,
          attn_mask=attn_mask,
          segment_ids=segment_ids,
      )
      self.assertTrue(kernel_called)
      self.assertEqual(out.shape, (b, t, config.embed_dim))
      self.assertFalse(jnp.isnan(out).any())

  def test_flash_attention_single_without_segment_ids(self):
    """Verify _flash_attention_single execution path without segment_ids."""
    config = model_lib.ModelConfig.gemma4_e2b()
    config.use_flash_attention = True
    config.flash_attention_block_size = 16
    attn = attention_lib.Attention(
        config=config,
        attn_type=model_lib.AttentionType.GLOBAL,
        rngs=nnx.Rngs(0),
    )
    b, t = 2, 16
    x = jnp.zeros((b, t, config.embed_dim))
    segment_pos = jnp.zeros((b, t), dtype=jnp.int32)
    attn_mask = jnp.ones((b, t, t), dtype=jnp.bool_)

    called_with_segments = []

    def mock_kernel(q_in, k_in, v_in, segment_ids=None):
      called_with_segments.append(segment_ids)
      return jnp.zeros_like(q_in)

    devices = np.array(jax.devices()[:1]).reshape(1, 1)
    mesh = jax.sharding.Mesh(devices, ('fsdp', 'tp'))

    with mesh, mock.patch.object(
        attn, '_make_splash_kernel', return_value=(mock_kernel, None)
    ):
      new_cache, out, (k_proj, v_proj, *_) = attn.block(
          x,
          segment_pos,
          cache=None,
          attn_mask=attn_mask,
          segment_ids=None,
      )
      self.assertEqual(called_with_segments, [None])
      self.assertEqual(out.shape, (b, t, config.embed_dim))
      self.assertEqual(k_proj.shape, (b, t, attn.num_kv_heads, attn.head_dim))
      self.assertEqual(v_proj.shape, (b, t, attn.num_kv_heads, attn.head_dim))
      self.assertFalse(jnp.isnan(out).any())

  @parameterized.named_parameters(
      dict(
          testcase_name='exact_boundary',
          prefix_length=128,
          cache_len=1024,
          boundaries=(0, 128, 256),
          expected=128,
      ),
      dict(
          testcase_name='in_between_rounds_up_to_next_boundary',
          prefix_length=100,
          cache_len=1024,
          boundaries=(0, 128, 256),
          expected=128,
      ),
      dict(
          testcase_name='overflow_past_boundaries_falls_back_to_cache_len',
          prefix_length=500,
          cache_len=1024,
          boundaries=(0, 128, 256),
          expected=1024,
      ),
      dict(
          testcase_name='beyond_cache_len_clamped_to_cache_len',
          prefix_length=2000,
          cache_len=1024,
          boundaries=(0, 128, 256),
          expected=1024,
      ),
      dict(
          testcase_name='boundary_exceeds_cache_len_clamped_to_cache_len',
          prefix_length=300,
          cache_len=256,
          boundaries=(0, 128, 512),
          expected=256,
      ),
      dict(
          testcase_name='empty_boundaries_ladder_falls_back_to_cache_len',
          prefix_length=100,
          cache_len=256,
          boundaries=(),
          expected=256,
      ),
  )
  def test_bucket_prefix_length(
      self, prefix_length, cache_len, boundaries, expected
  ):
    self.assertEqual(
        config_lib._bucket_prefix_length(prefix_length, cache_len, boundaries),
        expected,
    )

  def test_maybe_bucket_prefix_length(self):
    cache = {'v': jnp.zeros((1, 1024, 1, 64))}
    boundaries = (0, 128, 256)

    # Chunked prefill buckets prefix_length using cache_len.
    self.assertEqual(
        config_lib._maybe_bucket_prefix_length(
            100, cache, is_chunked_prefill=True, boundaries=boundaries
        ),
        128,
    )

    # Non-chunked prefill bypasses bucketing.
    self.assertEqual(
        config_lib._maybe_bucket_prefix_length(
            100, cache, is_chunked_prefill=False, boundaries=boundaries
        ),
        100,
    )

    # prefix_length <= 0 bypasses bucketing.
    self.assertEqual(
        config_lib._maybe_bucket_prefix_length(
            0, cache, is_chunked_prefill=True, boundaries=boundaries
        ),
        0,
    )

    # Empty boundaries ladder bypasses bucketing.
    self.assertEqual(
        config_lib._maybe_bucket_prefix_length(
            100, cache, is_chunked_prefill=True, boundaries=()
        ),
        100,
    )

  def test_bucket_generation_ladders(self):
    # pow2_buckets generates (0, 128, 256, ..., max_len)
    pow2 = config_lib.pow2_buckets(max_len=1024)
    self.assertEqual(pow2, (0, 128, 256, 512, 1024))
    pow2_default = config_lib.pow2_buckets()
    self.assertEqual(pow2_default[0], 0)
    self.assertEqual(pow2_default[1], 128)
    self.assertEqual(pow2_default[-1], 131072)

    # linear_buckets generates (0, step, 2*step, ..., max_len)
    linear = config_lib.linear_buckets(step=256, max_len=1024)
    self.assertEqual(linear, (0, 256, 512, 768, 1024))
    linear_default = config_lib.linear_buckets()
    self.assertEqual(linear_default[0], 0)
    self.assertEqual(linear_default[1], 512)
    self.assertEqual(linear_default[-1], 131072)

    # When max_len is not an exact multiple of step, or step=1, range upper bound
    # must strictly equal max_len (kills range(0, max_len + 1 + 1, step) mutant at config.py:91).
    self.assertEqual(
        config_lib.linear_buckets(step=1, max_len=5), (0, 1, 2, 3, 4, 5)
    )
    self.assertEqual(config_lib.linear_buckets(step=3, max_len=8), (0, 3, 6))

  def test_update_cache_prefill_end_index_advances_by_max_real_tokens(self):
    config = model_lib.ModelConfig.gemma4_e2b()
    config.sliding_window_size = 16
    config.use_sliding_window_kv_cache = True
    attn = attention_lib.Attention(
        config=config,
        attn_type=model_lib.AttentionType.GLOBAL,
        rngs=nnx.Rngs(0),
    )
    b, seq_len, cache_len = 2, 10, 16
    kh, d = config.num_kv_heads, config.head_dim
    key_proj = jnp.zeros((b, seq_len, kh, d))
    value_proj = jnp.zeros((b, seq_len, kh, d))

    # Case 1: Ragged input_mask. Row 0 has 7 tokens, row 1 has 4 tokens.
    # Batch-max real tokens = 7. end_index should advance by 7.
    input_mask = jnp.array(
        [[1] * 7 + [0] * 3, [1] * 4 + [0] * 6], dtype=jnp.bool_
    )
    cache = {
        'k': jnp.zeros((b, cache_len, kh, d)),
        'v': jnp.zeros((b, cache_len, kh, d)),
        'end_index': jnp.array([5, 5], dtype=jnp.int32),
    }
    updated_cache, *_ = attn._update_cache_prefill(
        cache,
        key_proj,
        value_proj,
        seq_len=seq_len,
        is_chunked_prefill=True,
        prefix_length=5,
        input_mask=input_mask,
    )
    np.testing.assert_array_equal(
        updated_cache['end_index'], jnp.array([12, 12], dtype=jnp.int32)
    )

    # Case 2: input_mask is None. end_index should advance by seq_len (10).
    cache = {
        'k': jnp.zeros((b, cache_len, kh, d)),
        'v': jnp.zeros((b, cache_len, kh, d)),
        'end_index': jnp.array([5, 5], dtype=jnp.int32),
    }
    updated_cache_none, *_ = attn._update_cache_prefill(
        cache,
        key_proj,
        value_proj,
        seq_len=seq_len,
        is_chunked_prefill=True,
        prefix_length=5,
        input_mask=None,
    )
    np.testing.assert_array_equal(
        updated_cache_none['end_index'], jnp.array([15, 15], dtype=jnp.int32)
    )
    # Active bucketing when chunked prefill (kills config.py:114 mutant)
    mock_cache = {'v': jnp.zeros((1, 1024, 1, 1))}
    self.assertEqual(
        config_lib._maybe_bucket_prefix_length(
            100, mock_cache, is_chunked_prefill=True, boundaries=(0, 128, 256)
        ),
        128,
    )
    # Direct tests for bucket generation ladders (kills config.py:83, 84 mutants)
    self.assertEqual(config_lib.pow2_buckets(1024), (0, 128, 256, 512, 1024))
    self.assertEqual(
        config_lib.linear_buckets(step=256, max_len=1024),
        (0, 256, 512, 768, 1024),
    )

  def test_ragged_ring_buffer_decode_matches_unbatched_ground_truth(self):
    """Ragged LOCAL_SLIDING prefill+decode must match B=1 unpadded ground truth."""
    config = model_lib.ModelConfig.gemma4_e2b()
    config.use_sliding_window_kv_cache = True
    config.sliding_window_size = 4
    attn = attention_lib.Attention(
        config=config,
        attn_type=model_lib.AttentionType.LOCAL_SLIDING,
        rngs=nnx.Rngs(0),
    )
    cache_len, kh, d = 4, config.num_kv_heads, config.head_dim
    lens = jnp.array([12, 6], dtype=jnp.int32)  # row 1 gap = 6 > window (4)
    x_pre = jax.random.normal(jax.random.PRNGKey(0), (2, 12, config.embed_dim))
    x_dec = jax.random.normal(jax.random.PRNGKey(1), (2, 1, config.embed_dim))

    def _run(row: int | None) -> jnp.ndarray:
      idx = slice(None) if row is None else slice(row, row + 1)
      b, seq = (2, 12) if row is None else (1, int(lens[row]))
      in_mask = jnp.arange(seq)[None, :] < lens[idx, None]
      causal = jnp.tril(jnp.ones((b, seq, seq), dtype=jnp.bool_))
      cache = {
          'k': jnp.zeros((b, cache_len, kh, d)),
          'v': jnp.zeros((b, cache_len, kh, d)),
          'end_index': jnp.zeros((b,), dtype=jnp.int32),
      }
      cache, _, _ = attn.block(
          x_pre[idx, :seq],
          jnp.broadcast_to(jnp.arange(seq, dtype=jnp.int32), (b, seq)),
          cache=cache,
          attn_mask=causal & in_mask[:, None, :],
          input_mask=in_mask,
          is_chunked_prefill=True,
          force_eager=True,
      )
      dec_mask = jnp.zeros((b, 1, seq + 1), dtype=jnp.bool_)
      dec_mask = dec_mask.at[:, 0, :seq].set(in_mask)
      dec_mask = dec_mask.at[:, 0, seq].set(True)
      _, out, _ = attn.block(
          x_dec[idx],
          lens[idx, None],
          cache=cache,
          attn_mask=dec_mask,
          force_eager=True,
      )
      return out

    ragged_out = _run(None)
    for r in range(len(lens)):
      np.testing.assert_allclose(
          ragged_out[r : r + 1],
          _run(r),
          atol=1e-2,
          rtol=1e-2,
          err_msg=f'Row {r} diverged from batch=1 ground truth!',
      )

  @parameterized.named_parameters(
      dict(
          testcase_name='sliding_window_within_window',
          attn_type=model_lib.AttentionType.LOCAL_SLIDING,
          sliding_window_size=16,
          prefix_len=8,
          suffix_len=4,
      ),
      dict(
          testcase_name='sliding_window_at_window_boundary',
          attn_type=model_lib.AttentionType.LOCAL_SLIDING,
          sliding_window_size=16,
          prefix_len=16,
          suffix_len=4,
      ),
      dict(
          testcase_name='sliding_window_beyond_window_wrapped',
          attn_type=model_lib.AttentionType.LOCAL_SLIDING,
          sliding_window_size=16,
          prefix_len=24,
          suffix_len=4,
      ),
      dict(
          testcase_name='global_prefix_reuse',
          attn_type=model_lib.AttentionType.GLOBAL,
          sliding_window_size=None,
          prefix_len=24,
          suffix_len=4,
      ),
  )
  def test_chunked_prefill_prefix_reuse_matches_full_prefill(
      self, attn_type, sliding_window_size, prefix_len, suffix_len
  ):
    config = model_lib.ModelConfig.gemma4_e2b()
    config.sliding_window_size = sliding_window_size
    config.use_sliding_window_kv_cache = True
    config.use_flash_attention = False

    attn = attention_lib.Attention(
        config=config,
        attn_type=attn_type,
        rngs=nnx.Rngs(0),
    )

    b = 2
    total_len = prefix_len + suffix_len
    x_full = jax.random.normal(
        jax.random.PRNGKey(0), (b, total_len, config.embed_dim)
    )
    pos_full = jnp.broadcast_to(
        jnp.arange(total_len, dtype=jnp.int32)[None, :], (b, total_len)
    )
    mask_full = jnp.tril(jnp.ones((b, total_len, total_len), dtype=jnp.bool_))

    # 1. Full monolithic prefill
    max_seq_len = total_len + 16
    cache_full = attn.init_cache(b, max_seq_len, dtype=jnp.float32)
    cache_full, out_full, _ = attn.block(
        x_full,
        pos_full,
        cache=cache_full,
        attn_mask=mask_full,
        is_chunked_prefill=False,
    )

    # 2. Chunked prefill: Chunk 1 (prefix)
    cache_chunked = attn.init_cache(b, max_seq_len, dtype=jnp.float32)
    x_prefix = x_full[:, :prefix_len, :]
    pos_prefix = pos_full[:, :prefix_len]
    mask_prefix = jnp.tril(
        jnp.ones((b, prefix_len, prefix_len), dtype=jnp.bool_)
    )
    cache_chunked, _, _ = attn.block(
        x_prefix,
        pos_prefix,
        cache=cache_chunked,
        attn_mask=mask_prefix,
        is_chunked_prefill=True,
        prefix_length=0,
        force_eager=True,
    )

    # 3. Chunked prefill: Chunk 2 (suffix reusing prefix cache)
    x_suffix = x_full[:, prefix_len:, :]
    pos_suffix = pos_full[:, prefix_len:]
    mask_suffix = mask_full[:, prefix_len:, :]
    cache_chunked, out_suffix_chunked, _ = attn.block(
        x_suffix,
        pos_suffix,
        cache=cache_chunked,
        attn_mask=mask_suffix,
        is_chunked_prefill=True,
        prefix_length=prefix_len,
        force_eager=True,
    )

    # Verify suffix representation matches the suffix region of monolithic prefill
    out_suffix_expected = out_full[:, prefix_len:, :]
    np.testing.assert_allclose(
        out_suffix_chunked, out_suffix_expected, atol=2e-5, rtol=1e-4
    )

    # 4. Decode step at position total_len
    x_decode = jax.random.normal(
        jax.random.PRNGKey(1), (b, 1, config.embed_dim)
    )
    pos_decode = jnp.array([[total_len], [total_len]], dtype=jnp.int32)
    mask_decode = jnp.broadcast_to(
        (jnp.arange(max_seq_len)[None, None, :] <= total_len),
        (b, 1, max_seq_len),
    )

    cache_full, out_decode_full, _ = attn.block(
        x_decode,
        pos_decode,
        cache=cache_full,
        attn_mask=mask_decode,
        is_chunked_prefill=False,
    )
    cache_chunked, out_decode_chunked, _ = attn.block(
        x_decode,
        pos_decode,
        cache=cache_chunked,
        attn_mask=mask_decode,
        is_chunked_prefill=False,
    )
    np.testing.assert_allclose(
        out_decode_chunked, out_decode_full, atol=2e-5, rtol=1e-4
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='global_unpadded',
          attn_type=model_lib.AttentionType.GLOBAL,
          use_input_mask=False,
      ),
      dict(
          testcase_name='global_padded',
          attn_type=model_lib.AttentionType.GLOBAL,
          use_input_mask=True,
      ),
      dict(
          testcase_name='local_sliding_unpadded',
          attn_type=model_lib.AttentionType.LOCAL_SLIDING,
          use_input_mask=False,
      ),
      dict(
          testcase_name='local_sliding_padded',
          attn_type=model_lib.AttentionType.LOCAL_SLIDING,
          use_input_mask=True,
      ),
  )
  def test_split_attention_eager_fallback_concatenates_prefix(
      self, attn_type, use_input_mask
  ):
    config = model_lib.ModelConfig.gemma4_e2b()
    config.sliding_window_size = 16
    config.use_sliding_window_kv_cache = True

    # 1. Baseline: use_split_attention = False
    config_base = dataclasses.replace(
        config, use_flash_attention=False, use_split_attention=False
    )
    attn_base = attention_lib.Attention(
        config=config_base,
        attn_type=attn_type,
        rngs=nnx.Rngs(0),
    )

    b, prefix_len, suffix_len = 2, 8, 4
    cache_len = 16
    kh, d = attn_base.num_kv_heads, attn_base.head_dim

    cache_split = {
        'k': jax.random.normal(jax.random.PRNGKey(10), (b, cache_len, kh, d)),
        'v': jax.random.normal(jax.random.PRNGKey(11), (b, cache_len, kh, d)),
        'end_index': jnp.array([prefix_len, prefix_len], dtype=jnp.int32),
    }
    cache_base = jax.tree.map(jnp.copy, cache_split)

    x = jax.random.normal(
        jax.random.PRNGKey(0), (b, suffix_len, config.embed_dim)
    )
    segment_pos = jnp.broadcast_to(
        jnp.arange(prefix_len, prefix_len + suffix_len, dtype=jnp.int32)[
            None, :
        ],
        (b, suffix_len),
    )
    total_len = prefix_len + suffix_len
    attn_mask = jnp.ones((b, suffix_len, total_len), dtype=jnp.bool_)
    if use_input_mask:
      input_mask = jnp.ones((b, suffix_len), dtype=jnp.bool_)
      input_mask = input_mask.at[1, -1].set(False)
    else:
      input_mask = None
    _, out_base, layers_kvs_base = attn_base.block(
        x,
        segment_pos,
        cache=cache_base,
        attn_mask=attn_mask,
        is_chunked_prefill=True,
        prefix_length=prefix_len,
        input_mask=input_mask,
        force_eager=True,
    )

    # 2. Split attention with eager fallback: use_split_attention = True,
    # force_eager = True
    config_split = dataclasses.replace(
        config, use_flash_attention=True, use_split_attention=True
    )
    attn_split = attention_lib.Attention(
        config=config_split,
        attn_type=attn_type,
        rngs=nnx.Rngs(0),
    )
    _, out_split, layers_kvs_split = attn_split.block(
        x,
        segment_pos,
        cache=cache_split,
        attn_mask=attn_mask,
        is_chunked_prefill=True,
        prefix_length=prefix_len,
        input_mask=input_mask,
        force_eager=True,
    )

    np.testing.assert_allclose(out_split, out_base, atol=1e-5, rtol=1e-5)
    self.assertEqual(len(layers_kvs_split), 6)
    k_proj, v_proj, _, _, split_k, split_v = layers_kvs_split
    self.assertIsNone(split_k)
    self.assertIsNone(split_v)
    np.testing.assert_allclose(k_proj, layers_kvs_base[0], atol=1e-5)
    np.testing.assert_allclose(v_proj, layers_kvs_base[1], atol=1e-5)

  @parameterized.named_parameters(
      dict(
          testcase_name='global',
          attn_type=model_lib.AttentionType.GLOBAL,
      ),
      dict(
          testcase_name='local_sliding',
          attn_type=model_lib.AttentionType.LOCAL_SLIDING,
      ),
  )
  def test_follower_layer_split_attention_with_kv_shared_cache(self, attn_type):
    config = model_lib.ModelConfig.gemma4_e2b()
    config.sliding_window_size = 16
    config.use_sliding_window_kv_cache = True
    config.flash_attention_block_size = 8

    # 1. Baseline origin & follower (no split attention, force_eager=True)
    config_base = dataclasses.replace(
        config, use_flash_attention=False, use_split_attention=False
    )
    attn_base_origin = attention_lib.Attention(
        config=config_base, attn_type=attn_type, rngs=nnx.Rngs(0)
    )
    attn_base_follower = attention_lib.Attention(
        config=config_base, attn_type=attn_type, rngs=nnx.Rngs(1)
    )

    b, prefix_len, suffix_len = 2, 8, 8
    cache_len = 16
    kh, d = attn_base_origin.num_kv_heads, attn_base_origin.head_dim

    cache_split = {
        'k': jax.random.normal(jax.random.PRNGKey(10), (b, cache_len, kh, d)),
        'v': jax.random.normal(jax.random.PRNGKey(11), (b, cache_len, kh, d)),
        'end_index': jnp.array([prefix_len, prefix_len], dtype=jnp.int32),
    }
    cache_base = jax.tree.map(jnp.copy, cache_split)

    x = jax.random.normal(
        jax.random.PRNGKey(0), (b, suffix_len, config.embed_dim)
    )
    segment_pos = jnp.broadcast_to(
        jnp.arange(prefix_len, prefix_len + suffix_len, dtype=jnp.int32)[
            None, :
        ],
        (b, suffix_len),
    )
    total_len = prefix_len + suffix_len
    attn_mask = jnp.ones((b, suffix_len, total_len), dtype=jnp.bool_)
    _, _, layers_kvs_base_origin = attn_base_origin.block(
        x,
        segment_pos,
        cache=cache_base,
        attn_mask=attn_mask,
        is_chunked_prefill=True,
        prefix_length=prefix_len,
        force_eager=True,
    )
    kv_shared_cache_base = {
        'k': layers_kvs_base_origin[0],
        'v': layers_kvs_base_origin[1],
        'valid_mask': layers_kvs_base_origin[2],
        'prior_end_index': layers_kvs_base_origin[3],
    }
    _, out_follower_base, _ = attn_base_follower.block(
        x,
        segment_pos,
        cache=None,
        attn_mask=attn_mask,
        kv_shared_cache=kv_shared_cache_base,
        is_chunked_prefill=True,
        prefix_length=prefix_len,
        force_eager=True,
    )

    # 2. Split attention origin layer -> obtain 6-element tuple
    config_split = dataclasses.replace(
        config, use_flash_attention=True, use_split_attention=True
    )
    attn_origin = attention_lib.Attention(
        config=config_split, attn_type=attn_type, rngs=nnx.Rngs(0)
    )
    attn_follower = attention_lib.Attention(
        config=config_split, attn_type=attn_type, rngs=nnx.Rngs(1)
    )

    def mock_kernel(q_in, k_in, v_in, segment_ids=None):
      out = jnp.zeros_like(q_in)
      lse = jnp.zeros(q_in.shape[:-1], dtype=jnp.float32)
      return out, (lse,)

    devices = np.array(jax.devices()[:1]).reshape(1, 1)
    mesh = jax.sharding.Mesh(devices, ('fsdp', 'tp'))

    with mesh, mock.patch.object(
        attn_origin, '_make_splash_kernel', return_value=(mock_kernel, None)
    ):
      _, _, layers_kvs_origin = attn_origin.block(
          x,
          segment_pos,
          cache=cache_split,
          attn_mask=attn_mask,
          is_chunked_prefill=True,
          prefix_length=prefix_len,
          force_eager=False,
      )

    self.assertEqual(len(layers_kvs_origin), 6)
    shared_k, shared_v, shared_mask, origin_end_idx, split_k, split_v = (
        layers_kvs_origin
    )
    self.assertIsNotNone(split_k)
    self.assertIsNotNone(split_v)

    kv_shared_cache = {
        'k': shared_k,
        'v': shared_v,
        'valid_mask': shared_mask,
        'prior_end_index': origin_end_idx,
        'split_prefix_k': split_k,
        'split_prefix_v': split_v,
    }

    # 3. Follower layer with Flash Split execution (zero-copy split attention)
    with mesh, mock.patch.object(
        attn_follower, '_make_splash_kernel', return_value=(mock_kernel, None)
    ), mock.patch.object(
        attn_follower,
        '_flash_attention_split',
        wraps=attn_follower._flash_attention_split,
    ) as mock_flash_split:
      _, out_follower_flash, layers_kvs_follower_flash = attn_follower.block(
          x,
          segment_pos,
          cache=None,
          attn_mask=attn_mask,
          kv_shared_cache=kv_shared_cache,
          is_chunked_prefill=True,
          prefix_length=prefix_len,
          force_eager=False,
      )
      mock_flash_split.assert_called_once()
      self.assertEqual(
          out_follower_flash.shape, (b, suffix_len, config.embed_dim)
      )
      self.assertFalse(jnp.isnan(out_follower_flash).any())
      self.assertIsNotNone(layers_kvs_follower_flash[4])
      self.assertIsNotNone(layers_kvs_follower_flash[5])

    # 4. Follower layer with Asymmetric Fallback (force_eager = True)
    _, out_follower_eager, layers_kvs_follower_eager = attn_follower.block(
        x,
        segment_pos,
        cache=None,
        attn_mask=attn_mask,
        kv_shared_cache=kv_shared_cache,
        is_chunked_prefill=True,
        prefix_length=prefix_len,
        force_eager=True,
    )
    self.assertIsNone(layers_kvs_follower_eager[4])
    self.assertIsNone(layers_kvs_follower_eager[5])
    np.testing.assert_allclose(
        out_follower_eager, out_follower_base, atol=1e-5, rtol=1e-5
    )

  def test_merge_split_attention_fully_masked_nan_safe(self):
    """Verifies _merge_split_attention avoids NaNs when partitions are fully masked."""
    b, n, t, h = 2, 2, 4, 8
    lse_prefix = jnp.array([[[-jnp.inf, -jnp.inf, -jnp.inf, 1.0]] * n] * b)
    lse_suffix = jnp.array([[[-jnp.inf, -jnp.inf, 0.0, 1.0]] * n] * b)

    out_prefix = jnp.ones((b, n, t, h), dtype=jnp.float32)
    out_suffix = jnp.ones((b, n, t, h), dtype=jnp.float32) * 2.0

    out_prefix = jnp.where(
        lse_prefix[..., None] == -jnp.inf, jnp.nan, out_prefix
    )
    out_suffix = jnp.where(
        lse_suffix[..., None] == -jnp.inf, jnp.nan, out_suffix
    )

    encoded = attention_lib._merge_split_attention(
        out_prefix,
        lse_prefix,
        out_suffix,
        lse_suffix,
        out_dtype=jnp.float32,
    )

    self.assertFalse(jnp.isnan(encoded).any())
    np.testing.assert_array_equal(
        encoded[:, :, :2, :], np.zeros((b, n, 2, h), dtype=np.float32)
    )

  def test_update_cache_prefill_3d_input_mask_axis_safety(self):
    """Verifies _update_cache_prefill with 3D input mask (kills axis=1 mutant at attention.py:452)."""
    config = model_lib.ModelConfig.gemma4_e2b()
    attn = attention_lib.Attention(
        config=config,
        attn_type=model_lib.AttentionType.GLOBAL,
        rngs=nnx.Rngs(0),
    )
    b, kh, d = 2, config.num_kv_heads, config.head_dim
    cache = {
        'k': jnp.zeros((b, 16, kh, d)),
        'v': jnp.zeros((b, 16, kh, d)),
        'end_index': jnp.zeros((b,), dtype=jnp.int32),
    }
    k_proj = jnp.zeros((b, 4, kh, d))
    v_proj = jnp.zeros((b, 4, kh, d))
    # 3D input mask: shape (2, 1, 4).
    # Row 0 has 3 valid tokens, Row 1 has 2 valid tokens.
    input_mask_3d = jnp.array(
        [[[True, True, True, False]], [[True, True, False, False]]]
    )
    updated_cache, *_ = attn._update_cache_prefill(
        cache,
        k_proj,
        v_proj,
        seq_len=4,
        is_chunked_prefill=True,
        prefix_length=0,
        input_mask=input_mask_3d,
    )
    # Under axis=-1: max real-token count across batch is 3.
    # Under mutant axis=1: sums over singleton dimension yielding max 1.
    self.assertEqual(int(updated_cache['end_index'][0]), 3)

  def test_eager_attention_non_chunked_prefill_sliding_window_offset(self):
    """Verifies sliding window offset in standard prefill when kv_len != q_len (kills -offset mutant at attention.py:1399)."""
    config = model_lib.ModelConfig.gemma4_e2b()
    config.sliding_window_size = 3
    attn = attention_lib.Attention(
        config=config,
        attn_type=model_lib.AttentionType.LOCAL_SLIDING,
        rngs=nnx.Rngs(0),
    )
    b, h, d = 1, config.num_heads, config.head_dim
    kh = config.num_kv_heads
    q_len, kv_len = 2, 6
    query_proj = jnp.ones((b, q_len, h, d))
    key_proj = jnp.ones((b, kv_len, kh, d))
    value_proj = jnp.arange(kv_len, dtype=jnp.float32)[None, :, None, None]
    value_proj = jnp.broadcast_to(value_proj, (b, kv_len, kh, d))
    attn_mask = jnp.ones((b, q_len, kv_len), dtype=jnp.bool_)
    segment_pos = jnp.array([[4, 5]])  # segment_pos.shape[1] == 2 > 1

    out = attn._eager_attention(
        query_proj,
        key_proj,
        value_proj,
        attn_mask=attn_mask,
        segment_pos=segment_pos,
        cache=None,
        kv_shared_cache=None,
        seq_len=q_len,
        is_chunked_prefill=False,
    )
    self.assertFalse(jnp.isnan(out).any())
    # Under offset = 4, positions 2, 3, 4, 5 are within sliding window [2, 5].
    # Their values in value_proj are 2, 3, 4, 5 (mean 3.5).
    # Under mutated offset = -4, sliding mask is all zeros, failing this assertion.
    np.testing.assert_allclose(out[0, 0, 0, 0], 3.5, atol=1e-3)

  def test_read_cache_prefill_prefix_length_zero_returns_none_splits(self):
    """Verifies _read_cache_prefill returns None for split KV when prefix_length=0 (kills mutant at attention.py:487)."""
    config = model_lib.ModelConfig.gemma4_e2b()
    config.use_split_attention = True
    attn = attention_lib.Attention(
        config=config,
        attn_type=model_lib.AttentionType.GLOBAL,
        rngs=nnx.Rngs(0),
    )
    b, kh, d = 1, config.num_kv_heads, config.head_dim
    cache = {
        'k': jnp.zeros((b, 16, kh, d)),
        'v': jnp.zeros((b, 16, kh, d)),
        'end_index': jnp.zeros((b,), dtype=jnp.int32),
    }
    key_proj = jnp.zeros((b, 4, kh, d))
    value_proj = jnp.zeros((b, 4, kh, d))
    prior_end_index = jnp.array([0])

    _, _, kv_valid_mask, split_k, split_v = attn._read_prefix_kv(
        cache,
        key_proj,
        value_proj,
        seq_len=4,
        is_chunked_prefill=True,
        prefix_length=0,
        prior_end_index=prior_end_index,
    )
    self.assertIsNone(split_k)
    self.assertIsNone(split_v)
    self.assertIsNone(kv_valid_mask)

  def test_eager_attention_non_chunked_prefill_does_not_build_chunked_mask(self):
    """Verifies eager attention does not call _build_chunked_prefill_mask when is_chunked_prefill=False (kills mutant at attention.py:1329)."""
    config = model_lib.ModelConfig.gemma4_e2b()
    attn = attention_lib.Attention(
        config=config,
        attn_type=model_lib.AttentionType.GLOBAL,
        rngs=nnx.Rngs(0),
    )
    b, h, d = 1, config.num_heads, config.head_dim
    kh = config.num_kv_heads
    q_len, kv_len = 2, 6
    query_proj = jnp.ones((b, q_len, h, d))
    key_proj = jnp.ones((b, kv_len, kh, d))
    value_proj = jnp.ones((b, kv_len, kh, d))
    attn_mask = jnp.ones((b, q_len, kv_len), dtype=jnp.bool_)
    segment_pos = jnp.zeros((b, q_len), dtype=jnp.int32)

    with mock.patch.object(
        attn,
        '_build_chunked_prefill_mask',
        wraps=attn._build_chunked_prefill_mask,
    ) as mock_build_mask:
      out = attn._eager_attention(
          query_proj,
          key_proj,
          value_proj,
          attn_mask=attn_mask,
          segment_pos=segment_pos,
          cache=None,
          kv_shared_cache=None,
          seq_len=q_len,
          is_chunked_prefill=False,
      )
      mock_build_mask.assert_not_called()
      self.assertEqual(out.shape, (b, q_len, h, d))


if __name__ == '__main__':
  absltest.main()
