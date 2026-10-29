"""A user turn that only names a language switches before the LLM runs."""

import pytest
from pipecat.frames.frames import (
    LLMContextFrame,
    STTUpdateSettingsFrame,
    TTSUpdateSettingsFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.tests.utils import run_test

from call_processing.services.common.direct_language_switch import (
    DirectLanguageSwitchProcessor,
    LanguageSwitchConfig,
    match_language_choice,
)
from call_processing.services.common.prompts import language_instruction

BASE_PROMPT = 'You help the user.'
SUPPORTED = ['en', 'hi', 'ta']


# ------------------------------------------------------------------- config


def test_direct_switch_is_off_by_default():
    assert LanguageSwitchConfig.from_agent_config({}).direct_switch is False


def test_direct_switch_can_be_turned_on():
    agent = {'conversation_config': {'language_switch': {'direct_switch': True}}}

    assert LanguageSwitchConfig.from_agent_config(agent).direct_switch is True


def test_non_boolean_direct_switch_falls_back_to_the_default():
    agent = {'conversation_config': {'language_switch': {'direct_switch': 'true'}}}

    assert LanguageSwitchConfig.from_agent_config(agent).direct_switch is False


# ------------------------------------------------------------------ matching


@pytest.mark.parametrize(
    'text, language',
    [
        ('Hindi.', 'hi'),
        ('hindi', 'hi'),
        ('हिंदी।', 'hi'),
        ('हिन्दी', 'hi'),
        ('Hindi please', 'hi'),
        ('Hindi mein', 'hi'),
        ('हिंदी में', 'hi'),
        ('Hindi. Hindi.', 'hi'),
        ('Yes, English.', 'en'),
        ('Speak Tamil', 'ta'),
        ('Continue in Hindi please.', 'hi'),
    ],
)
def test_a_bare_language_choice_is_matched(text, language):
    assert match_language_choice(text, SUPPORTED) == language


@pytest.mark.parametrize(
    'text',
    [
        'Can we talk in Hindi?',
        'Not Hindi',
        'I need a loan in Hindi',
        'Hindi or English',
        'Kannada',  # not supported by this agent
        'Indy',  # a mishearing is left to the LLM's judgment
        '',
        '...',
        'please',
    ],
)
def test_anything_more_than_a_choice_is_left_to_the_llm(text):
    assert match_language_choice(text, SUPPORTED) is None


def test_multi_word_language_names_match():
    assert match_language_choice('Bahasa Indonesia', ['en', 'id']) == 'id'


def test_a_name_shared_by_two_supported_languages_is_left_to_the_llm():
    assert match_language_choice('Filipino', ['en', 'fil', 'tl']) is None
    assert match_language_choice('Filipino', ['en', 'fil']) == 'fil'


# ----------------------------------------------------------------- processor


class FakeTask:
    def __init__(self):
        self.queued = []

    async def queue_frame(self, frame):
        self.queued.append(frame)


class Harness:
    def __init__(self, user_text: str, enabled: bool = True, current: str = 'en'):
        self.context = LLMContext(
            [
                {
                    'role': 'system',
                    'content': f'{language_instruction(current)}\n\n{BASE_PROMPT}',
                },
                {'role': 'assistant', 'content': 'Which language would you like?'},
                {'role': 'user', 'content': user_text},
            ]
        )
        self.task = FakeTask()
        self.selected = []
        self.state = {
            'current_language': current,
            'switch_count': 0,
            'original_system_prompt': BASE_PROMPT,
        }

        async def on_selected(language):
            self.selected.append(language)

        self.processor = DirectLanguageSwitchProcessor(
            enabled=enabled,
            supported_languages=SUPPORTED,
            language_state=self.state,
            task_container={'task': self.task},
            tts_provider='sarvam',
            stt_provider='sarvam',
            tts_voice_ids={'hi': 'v-hi'},
            on_language_selected=on_selected,
        )

    async def run(self):
        down, _ = await run_test(
            self.processor, frames_to_send=[LLMContextFrame(self.context)]
        )
        return [type(f) for f in down]

    def messages(self):
        return self.context.get_messages()


async def test_a_bare_language_name_switches_before_the_llm_runs():
    h = Harness('Hindi.')

    down = await h.run()

    # The voice change reaches the TTS ahead of the context the LLM answers
    assert down == [TTSUpdateSettingsFrame, LLMContextFrame]
    assert [type(f) for f in h.task.queued] == [STTUpdateSettingsFrame]
    assert h.messages()[0]['content'] == (
        f"{language_instruction('hi')}\n\n{BASE_PROMPT}"
    )
    assert h.state['current_language'] == 'hi'
    assert h.state['switch_count'] == 1
    assert h.selected == ['hi']
    # The LLM is told the switch is done, so it answers instead of calling the tool
    note = h.messages()[-1]
    assert note['role'] == 'developer'
    assert 'Hindi' in note['content']


async def test_choosing_the_current_language_adds_the_note_without_switching():
    h = Harness('English', current='en')

    down = await h.run()

    assert down == [LLMContextFrame]
    assert h.task.queued == []
    assert h.state['switch_count'] == 0
    assert h.selected == ['en']
    assert h.messages()[-1]['role'] == 'developer'


async def test_off_passes_the_turn_to_the_llm_untouched():
    h = Harness('Hindi.', enabled=False)

    down = await h.run()

    assert down == [LLMContextFrame]
    assert h.task.queued == []
    assert h.state['current_language'] == 'en'
    assert h.messages()[-1] == {'role': 'user', 'content': 'Hindi.'}


async def test_a_sentence_about_a_language_goes_to_the_llm_tool():
    h = Harness('Can we talk in Hindi?')

    down = await h.run()

    assert down == [LLMContextFrame]
    assert h.task.queued == []
    assert h.state['current_language'] == 'en'
    assert h.selected == []


async def test_a_run_that_does_not_end_on_a_user_message_is_left_alone():
    h = Harness('Hindi.')
    h.context.add_message({'role': 'developer', 'content': 'Check in on the user.'})

    down = await h.run()

    assert down == [LLMContextFrame]
    assert h.state['current_language'] == 'en'


async def test_without_a_pipeline_task_nothing_is_switched():
    h = Harness('Hindi.')
    h.processor._task_container = {'task': None}

    down = await h.run()

    assert down == [LLMContextFrame]
    assert h.state['current_language'] == 'en'
    assert h.messages()[-1]['role'] == 'user'
