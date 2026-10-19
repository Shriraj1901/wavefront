"""
User turn and mute strategies for voice conversations
"""

import time
from typing import Callable, List

from pipecat.frames.frames import Frame, InterimTranscriptionFrame, TranscriptionFrame
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.turns.types import ProcessFrameResult
from pipecat.turns.user_mute import BaseUserMuteStrategy
from pipecat.turns.user_start import TranscriptionUserTurnStartStrategy

from call_processing.log.logger import logger


class GuardedTranscriptionUserTurnStartStrategy(TranscriptionUserTurnStartStrategy):
    """Start a user turn from a transcript, unless it is of speech made while muted.

    Muting happens in the user aggregator, downstream of STT, so the final
    transcript of something said while muted (over the greeting, during a tool
    call) can reach the aggregator after the user is unmuted. The plain
    transcription strategy would start a turn from it, and the bot would answer
    stale speech. Inside the grace window after unmute, a final transcript that
    no turn has claimed is dropped from the aggregation instead.
    """

    def __init__(self, *, grace_secs: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self._grace_secs = grace_secs
        self._unmuted_at: float | None = None
        self._turn_started_since_unmute = True
        self._turn_active = False

    def notify_unmuted(self) -> None:
        """Open the grace window; call synchronously when the user is unmuted.

        A turn still open across the mute keeps its transcripts, so the window
        only opens when no turn is in progress.
        """
        if self._turn_active:
            return
        self._unmuted_at = time.monotonic()
        self._turn_started_since_unmute = False

    async def handle_user_turn_started(self):
        self._turn_active = True
        self._turn_started_since_unmute = True
        await super().handle_user_turn_started()

    async def handle_user_turn_stopped(self):
        self._turn_active = False
        await super().handle_user_turn_stopped()

    def _in_grace_window(self) -> bool:
        if self._turn_started_since_unmute or self._unmuted_at is None:
            return False
        return time.monotonic() - self._unmuted_at < self._grace_secs

    async def process_frame(self, frame: Frame) -> ProcessFrameResult:
        is_transcript = isinstance(
            frame, (TranscriptionFrame, InterimTranscriptionFrame)
        )
        if is_transcript and self._in_grace_window():
            # An interim starts nothing; a final is dropped from the aggregation.
            if isinstance(frame, TranscriptionFrame):
                logger.debug('Dropping a transcript of speech made while muted')
                await self.trigger_reset_aggregation()
            return ProcessFrameResult.CONTINUE
        return await super().process_frame(frame)


class UnmuteNotifyingUserMuteStrategy(BaseUserMuteStrategy):
    """Mute while any wrapped strategy mutes; report the unmute as it happens.

    The aggregator's on_user_mute_stopped event runs its handlers in a separate
    task, so a frame queued right behind the unmuting one can be processed
    first. The aggregator evaluates mute strategies inline on every frame, so
    calling on_unmuted here runs before any later frame reaches the turn
    strategies.
    """

    def __init__(
        self,
        strategies: List[BaseUserMuteStrategy],
        on_unmuted: Callable[[], None],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._strategies = strategies
        self._on_unmuted = on_unmuted
        self._muted = False

    @property
    def strategies(self) -> List[BaseUserMuteStrategy]:
        return self._strategies

    async def setup(self, setup: FrameProcessorSetup):
        await super().setup(setup)
        for strategy in self._strategies:
            await strategy.setup(setup)

    async def cleanup(self):
        for strategy in self._strategies:
            await strategy.cleanup()
        await super().cleanup()

    async def process_frame(self, frame: Frame) -> bool:
        # Every strategy sees every frame: each tracks its own state.
        muted = False
        for strategy in self._strategies:
            muted |= await strategy.process_frame(frame)
        if self._muted and not muted:
            self._on_unmuted()
        self._muted = muted
        return muted
