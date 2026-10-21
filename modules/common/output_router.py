"""Output Decision Module — Routes processed text to appropriate output modalities.

Based on User B's profile (deaf/blind/mute/normal), determines which
output channels to activate:
  - Text display (screen/captions)
  - Voice TTS (text-to-speech)  
"""


class OutputRouter:
    """Routes messages to the correct output modalities based on User B's role."""

    # Mapping of receiver role to output actions
    ROUTING_TABLE = {
        'deaf': {
            'show_text': True,
            'speak_text': False,    # Deaf user can't hear TTS
        },
        'blind': {
            'show_text': False,     # Blind user can't read screen
            'speak_text': True,     # Auto-read via TTS
        },
        'mute': {
            'show_text': True,      # Mute user can see text
            'speak_text': True,     # Mute user can hear TTS
        },
        'normal': {
            'show_text': True,
            'speak_text': True,
        }
    }

    def __init__(self, receiver_role):
        """
        Args:
            receiver_role: User B's role (deaf/blind/mute/normal).
        """
        self.receiver_role = receiver_role
        self.routing = self.ROUTING_TABLE.get(receiver_role, self.ROUTING_TABLE['normal'])

    def route(self, processed_text, source='text', glosses=None):
        """Determine outputs for a processed message.
        
        Args:
            processed_text: The LLM-processed English text.
            source: Input source ('sign', 'speech', 'text').
            glosses: Optional list of original sign glosses.
            
        Returns:
            dict: Output instructions with format:
                {
                    'text': str or None,
                    'tts_text': str or None,
                    'show_text': bool,
                    'speak_text': bool,
                }
        """
        result = {
            'text': processed_text if self.routing['show_text'] else None,
            'tts_text': processed_text if self.routing['speak_text'] else None,
            'show_text': self.routing['show_text'],
            'speak_text': self.routing['speak_text'],
        }
        return result

    def get_active_outputs(self):
        """Return list of active output modality names."""
        active = []
        if self.routing['show_text']:
            active.append('text')
        if self.routing['speak_text']:
            active.append('tts')
        return active
