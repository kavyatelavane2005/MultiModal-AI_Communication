"""Session Manager — Manages communication sessions between User A and User B.

Tracks user roles, determines available input/output modalities,
and manages session lifecycle (start → communicate → end).
"""


# Role types
ROLES = ['deaf', 'blind', 'mute', 'normal']

# Input modalities available per sender role
INPUT_MODES = {
    'deaf':   ['sign', 'text'],          # Can't hear/speak → uses sign + text
    'blind':  ['speech', 'text'],        # Can't see camera → uses speech + text
    'mute':   ['sign', 'text'],          # Can't speak → uses sign + text
    'normal': ['sign', 'speech', 'text'] # All modalities available
}

# Output modalities based on receiver role
OUTPUT_MODES = {
    'deaf':   ['text', 'avatar'],        # Can't hear → text display + sign avatar
    'blind':  ['tts'],                   # Can't see → voice only (auto-read)
    'mute':   ['text', 'tts'],           # Can hear + see → text + voice
    'normal': ['text', 'tts', 'avatar']  # All outputs
}


class Session:
    """Represents an active communication session."""

    def __init__(self, user_a_role, user_b_role):
        """
        Args:
            user_a_role: Role of sender (deaf/blind/mute/normal).
            user_b_role: Role of receiver (deaf/blind/mute/normal).
        """
        if user_a_role not in ROLES:
            raise ValueError(f"Invalid sender role: {user_a_role}")
        if user_b_role not in ROLES:
            raise ValueError(f"Invalid receiver role: {user_b_role}")

        self.user_a_role = user_a_role
        self.user_b_role = user_b_role
        self.active = True
        self.conversation_history = []

    @property
    def available_inputs(self):
        """Input modalities available to User A based on their role."""
        return INPUT_MODES[self.user_a_role]

    @property
    def output_modes(self):
        """Output modalities for User B based on their role."""
        return OUTPUT_MODES[self.user_b_role]

    def add_message(self, text, source, processed_text, outputs):
        """Add a message to conversation history.
        
        Args:
            text: Original input text.
            source: Input source ('sign', 'speech', 'text').
            processed_text: LLM-processed text.
            outputs: Dict of output actions taken.
        """
        self.conversation_history.append({
            'original': text,
            'source': source,
            'processed': processed_text,
            'outputs': outputs
        })

    def swap_roles(self):
        """Swap roles between User A and User B for two-way conversation turns."""
        self.user_a_role, self.user_b_role = self.user_b_role, self.user_a_role

    def end(self):
        """End the session."""
        self.active = False

    def to_dict(self):
        """Serialize session state for the client."""
        return {
            'user_a_role': self.user_a_role,
            'user_b_role': self.user_b_role,
            'available_inputs': self.available_inputs,
            'output_modes': self.output_modes,
            'active': self.active,
            'message_count': len(self.conversation_history)
        }
