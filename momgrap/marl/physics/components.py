"""Direct / reflected / aggregate comm-channel access for the MARL layer.

The decomposition itself lives in :mod:`momgrap.channels` so the core stays the
single source of truth and ``build_comm_channels`` remains a bit-for-bit wrapper
over :func:`build_comm_channel_components`. This module only re-exports the public
API under the ``momgrap.marl.physics`` namespace, which is where the MARL
observation builders (Task 09) import it from.

Component shapes (all complex, on ``cfg.device``)::

    direct     (N, S, K, M)   BS -> user direct row
    bs_ris     (N, S, L, M)   BS -> RIS channel
    ris_user   (S, K, L)      RIS -> user (MA-position independent)
    reflected  (N, S, K, M)   RIS-reflected row
    aggregate  (N, S, K, M)   direct + reflected  (== build_comm_channels)
"""

from __future__ import annotations

from ...channels import CommChannelComponents, build_comm_channel_components

__all__ = ["CommChannelComponents", "build_comm_channel_components"]
