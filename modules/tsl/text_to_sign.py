"""Text-to-Sign Language (TSL) Module.

Converts English sentences or speech into ASL gloss sequences
for sign language avatar animation and visualization.
"""
import re
from config.settings import ASL100_GLOSSES

# Common English stopwords to drop when converting English -> ASL Gloss
STOPWORDS = {
    'a', 'an', 'the', 'is', 'are', 'was', 'were', 'am', 'be', 'been', 'being',
    'to', 'of', 'and', 'in', 'that', 'have', 'has', 'had', 'do', 'does', 'did'
}

# Synonym mapping to existing 100 ASL glosses
SYNONYM_MAP = {
    'read': 'book',
    'reading': 'book',
    'water': 'drink',
    'beverage': 'drink',
    'pc': 'computer',
    'laptop': 'computer',
    'prior': 'before',
    'earlier': 'before',
    'seat': 'chair',
    'sit': 'chair',
    'walk': 'walk',
    'walking': 'walk',
    'run': 'go',
    'leave': 'go',
    'dress': 'clothes',
    'shirt': 'shirt',
    'clothing': 'clothes',
    'whoever': 'who',
    'sweet': 'candy',
    'relative': 'cousin',
    'good': 'fine',
    'great': 'fine',
    'assist': 'help',
    'support': 'help',
    'nope': 'no',
    'lean': 'thin',
    'slim': 'thin',
    'annual': 'year',
    'yeah': 'yes',
    'yep': 'yes',
    'every': 'all',
    'everyone': 'all',
    'everything': 'all',
    'done': 'finish',
    'completed': 'finish',
    'warm': 'hot',
    'love': 'like',
    'adore': 'like',
    'numerous': 'many',
    'lot': 'many',
    'mom': 'mother',
    'mama': 'mother',
    'currently': 'now',
    'today': 'now',
    'desk': 'table',
    'lady': 'woman',
    'female': 'woman',
    'sleep': 'bed',
    'puppy': 'dog',
    'canine': 'dog',
    'relatives': 'family',
    'grad': 'graduate',
    'cap': 'hat',
    'listen': 'hearing',
    'hear': 'hearing',
    'smooch': 'kiss',
    'talk': 'language',
    'speech': 'language',
    'afterwards': 'later',
    'male': 'man',
    'guy': 'man',
    'boy': 'man',
    'learn': 'study',
    'high': 'tall',
    'incorrect': 'wrong',
    'crash': 'accident',
    'fruit': 'apple',
    'modify': 'change',
    'alter': 'change',
    'physician': 'doctor',
    'medic': 'doctor',
    'consume': 'eat',
    'food': 'eat',
    'lunch': 'eat',
    'dinner': 'eat',
    'breakfast': 'eat',
    'fun': 'enjoy',
    'lose': 'forget',
    'provide': 'give',
    'offer': 'give',
    'final': 'last',
    'see': 'meet',
    'game': 'play',
    'academy': 'school',
    'college': 'school',
    'little': 'short',
    'clock': 'time',
    'hour': 'time',
    'desire': 'want',
    'wish': 'want',
    'labor': 'work',
    'job': 'work',
    'bball': 'basketball',
    'bday': 'birthday',
    'however': 'but',
    'town': 'city',
    'bake': 'cook',
    'choose': 'decide',
    'packed': 'full',
    'coat': 'jacket',
    'message': 'letter',
    'mail': 'letter',
    'pill': 'medicine',
    'drug': 'medicine',
    'require': 'need',
    'must': 'need',
    'draw': 'paint',
    'sheet': 'paper',
    'drag': 'pull',
    'correct': 'right',
    'identical': 'same',
    'speak': 'tell',
    'say': 'tell'
}


class TextToSign:
    """Translates English text into ASL gloss sequence."""

    def __init__(self, vocab=None):
        self.vocab = set(v.lower() for v in (vocab or ASL100_GLOSSES))

    def text_to_glosses(self, text):
        """Converts text into a list of valid ASL glosses.
        
        Args:
            text (str): Input English text
            
        Returns:
            list[str]: Sequence of recognized ASL glosses
        """
        if not text:
            return []

        # Tokenize and clean
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        glosses = []

        for word in words:
            # 1. Exact match in vocabulary
            if word in self.vocab:
                glosses.append(word)
                continue

            # 2. Synonym match
            if word in SYNONYM_MAP and SYNONYM_MAP[word] in self.vocab:
                glosses.append(SYNONYM_MAP[word])
                continue

            # 3. Simple stemming (e.g. drinks -> drink, books -> book, played -> play)
            stemmed = None
            if word.endswith('s') and word[:-1] in self.vocab:
                stemmed = word[:-1]
            elif word.endswith('ing') and word[:-3] in self.vocab:
                stemmed = word[:-3]
            elif word.endswith('ed') and word[:-2] in self.vocab:
                stemmed = word[:-2]

            if stemmed:
                glosses.append(stemmed)
                continue

            # If word is not a stopword and is at least 2 chars, keep it for fingerspelling
            if word not in STOPWORDS and len(word) >= 2:
                glosses.append(word)

        # Fallback: if no glosses matched, return non-stopword words or first few words
        if not glosses:
            glosses = [w for w in words if w not in STOPWORDS] or words[:3]

        return glosses
