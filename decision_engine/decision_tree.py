# decision_engine/decision_tree.py

class DecisionTree:
    def __init__(self):
        self.rules = {
            "LLM": {
                "subtasks": {
                    "code": ["TheStack", "CodeParrot", "BigCodeBench"],
                    "chat": ["ShareGPT", "OpenAssistant", "UltraChat"],
                    "translation": ["WMT", "ParaCrawl"],
                    "summarization": ["CNN/DailyMail", "XSUM"]
                }
            },

            "CV": {
                "subtasks": {
                    "detection": ["COCO", "Objects365", "OpenImages"],
                    "classification": ["ImageNet", "CIFAR-10", "CIFAR-100"],
                    "segmentation": ["ADE20K", "Cityscapes"]
                }
            },

            "Audio": {
                "subtasks": {
                    "speech_to_text": ["LibriSpeech", "CommonVoice"],
                    "speaker_id": ["VoxCeleb", "LibriSpeech"],
                    "audio_classification": ["ESC-50", "UrbanSound8K"]
                }
            },

            "Tabular": {
                "subtasks": {
                    "classification": ["UCI Adult", "Titanic"],
                    "regression": ["Boston Housing", "Diabetes"]
                }
            }
        }

    def get_subtasks(self, task_type):
        task = self.rules.get(task_type)
        if not task:
            return None
        return list(task["subtasks"].keys())

    def get_datasets(self, task_type, subtask):
        task = self.rules.get(task_type)
        if not task:
            return None
        return task["subtasks"].get(subtask)
