# decision_engine/decision_tree.py

class DecisionTree:
    """
    Дерево решений для фильтрации моделей и датасетов
    Версия: Прототип v1.0 (Tabular data focus)
    """
    
    def __init__(self):
        self.rules = {
            # Основной фокус: табличные данные (для прототипа)
            "Tabular": {
                "subtasks": {
                    "classification": {
                        "models": [
                            "LogisticRegression",
                            "RandomForestClassifier", 
                            "GradientBoostingClassifier",
                            "SVC",
                            "KNeighborsClassifier"
                        ],
                        "datasets": {
                            "sklearn": [
                                "load_iris",           # Классификация цветов
                                "load_wine",           # Классификация вин
                                "load_breast_cancer",  # Медицинская диагностика
                                "load_digits"          # Распознавание цифр
                            ],
                            "kaggle": [
                                "alexisbcook/titanic",                    # Выживаемость на Титанике
                                "uciml/mushroom-classification",          # Классификация грибов
                                "uciml/sms-spam-collection-dataset"       # Классификация спама
                            ]
                        },
                        "description": "Предсказание категориальной переменной (класса)"
                    },
                    "regression": {
                        "models": [
                            "LinearRegression",
                            "Ridge",
                            "Lasso",
                            "RandomForestRegressor",
                            "GradientBoostingRegressor"
                        ],
                        "datasets": {
                            "sklearn": [
                                "load_diabetes",       # Прогресс диабета
                                "california_housing"   # Цены на недвижимость
                            ],
                            "kaggle": [
                                "c/house-prices-advanced-regression-techniques",
                                "c/bike-sharing-demand"
                            ]
                        },
                        "description": "Предсказание числовой переменной"
                    }
                }
            },
            
            # Будущее расширение (пока не используется)
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
            }
        }

    def get_subtasks(self, task_type):
        """Возвращает список подзадач для типа задачи"""
        task = self.rules.get(task_type)
        if not task:
            return None
        return list(task["subtasks"].keys())

    def get_datasets(self, task_type, subtask, source="all"):
        """
        Возвращает список датасетов
        Совместимость со старым API + новая функциональность
        
        Args:
            task_type: Тип задачи (Tabular, CV, etc.)
            subtask: Подзадача (classification, regression)
            source: Источник ("all", "sklearn", "huggingface", "kaggle")
        """
        task = self.rules.get(task_type)
        if not task:
            return None
        
        subtask_data = task["subtasks"].get(subtask)
        
        # Для новой структуры Tabular (с несколькими источниками)
        if isinstance(subtask_data, dict) and "datasets" in subtask_data:
            datasets = subtask_data["datasets"]
            
            # Если datasets - это словарь источников
            if isinstance(datasets, dict):
                if source == "all":
                    # Объединить все источники
                    all_datasets = []
                    for src_datasets in datasets.values():
                        all_datasets.extend(src_datasets)
                    return all_datasets
                else:
                    # Вернуть конкретный источник
                    return datasets.get(source, [])
            else:
                # Старый формат - просто список
                return datasets
        
        # Для старой структуры (LLM, CV, Audio)
        return subtask_data

    def get_models(self, task_type, subtask):
        """
        НОВЫЙ: Возвращает список моделей для задачи
        """
        task = self.rules.get(task_type)
        if not task:
            return None
        
        subtask_data = task["subtasks"].get(subtask)
        
        if isinstance(subtask_data, dict):
            return subtask_data.get("models", [])
        
        return None

    def get_task_info(self, task_type, subtask):
        """
        НОВЫЙ: Возвращает полную информацию о задаче
        """
        task = self.rules.get(task_type)
        if not task:
            return None
        
        subtask_data = task["subtasks"].get(subtask)
        
        if isinstance(subtask_data, dict):
            return {
                "models": subtask_data.get("models", []),
                "datasets": subtask_data.get("datasets", []),
                "description": subtask_data.get("description", "")
            }
        
        return None

    def filter_by_criteria(self, task_type, criteria):
        """
        НОВЫЙ: Фильтрует варианты по критериям пользователя
        
        Пример criteria:
        {
            "fast_training": True,
            "interpretable": True,
            "small_data": True
        }
        """
        task = self.rules.get(task_type)
        if not task:
            return None
        
        results = {}
        for subtask_name, subtask_data in task["subtasks"].items():
            if isinstance(subtask_data, dict):
                models = subtask_data["models"]
                
                # Фильтрация по критериям
                if criteria.get("fast_training"):
                    # Простые и быстрые модели
                    models = [m for m in models 
                             if "Linear" in m or "Logistic" in m or "KNeighbors" in m]
                
                if criteria.get("interpretable"):
                    # Интерпретируемые модели
                    models = [m for m in models 
                             if "Linear" in m or "Logistic" in m or "Tree" in m]
                
                if criteria.get("high_accuracy"):
                    # Точные модели (ансамбли)
                    models = [m for m in models 
                             if "Forest" in m or "Boosting" in m]
                
                results[subtask_name] = {
                    "models": models if models else subtask_data["models"],
                    "datasets": subtask_data["datasets"]
                }
        
        return results