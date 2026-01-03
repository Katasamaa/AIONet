# decision_engine/llm_optimizer.py

import os
import json
import re
from openai import OpenAI

class LLMOptimizer:
    """
    Интеграция с OpenAI GPT для интеллектуального анализа задач ML
    Версия: Прототип v1.0
    """
    
    def __init__(self, api_key=None):
        """
        Инициализация клиента OpenAI
        
        Args:
            api_key: если None, берется из переменной окружения OPENAI_API_KEY
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "⚠️ OpenAI API ключ не найден!\n"
                "Создайте файл .env в корне проекта:\n"
                "OPENAI_API_KEY=ваш_ключ"
            )
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = "gpt-4o-mini"  # Дешевая модель, $5 бесплатно
        
        print(f"✅ LLM инициализирован: {self.model}")
    
    def _call_llm(self, prompt, max_tokens=1024, temperature=0.3):
        """
        Внутренний метод для вызова OpenAI API
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            raise Exception(f"Ошибка OpenAI API: {str(e)}")
    
    def parse_task(self, user_description):
        """
        Парсит описание задачи от пользователя в структурированный JSON
        
        Args:
            user_description (str): Описание задачи на естественном языке
            
        Returns:
            dict: Структурированные параметры задачи
            
        Example:
            >>> llm.parse_task("Хочу предсказывать цены на квартиры")
            {
                "task_type": "regression",
                "recommended_model": "RandomForestRegressor",
                ...
            }
        """
        
        prompt = f"""Проанализируй задачу машинного обучения и верни ТОЛЬКО валидный JSON.

Задача пользователя: "{user_description}"

Верни JSON строго в таком формате:
{{
    "task_type": "classification" или "regression",
    "data_description": "краткое описание данных из запроса",
    "recommended_model": "одна из моделей ниже",
    "reasoning": "почему выбрана эта модель (1-2 предложения)",
    "estimated_complexity": "low/medium/high",
    "key_features": ["список важных признаков если упомянуты, иначе []"],
    "target": "целевая переменная если упомянута, иначе null"
}}

Доступные модели для классификации:
- LogisticRegression (быстрая, интерпретируемая)
- RandomForestClassifier (точная, устойчивая)
- GradientBoostingClassifier (очень точная, медленная)
- SVC (для сложных границ)
- KNeighborsClassifier (простая, для малых данных)

Доступные модели для регрессии:
- LinearRegression (простая, быстрая)
- Ridge (регуляризованная линейная)
- Lasso (выбор признаков)
- RandomForestRegressor (нелинейная зависимость)
- GradientBoostingRegressor (очень точная)

ВАЖНО: 
- Если задача про категории/классы/типы → classification
- Если задача про числовые значения/цены/количество → regression
- Верни ТОЛЬКО JSON, без markdown и пояснений!"""

        try:
            response_text = self._call_llm(prompt, max_tokens=1024, temperature=0.3)
            
            # Очистка от markdown если GPT добавил
            response_text = re.sub(r'```json\s*', '', response_text)
            response_text = re.sub(r'```\s*', '', response_text)
            response_text = response_text.strip()
            
            # Парсинг JSON
            parsed = json.loads(response_text)
            
            # Валидация обязательных полей
            required_fields = ["task_type", "recommended_model"]
            for field in required_fields:
                if field not in parsed:
                    return {
                        "error": f"LLM не вернул обязательное поле: {field}",
                        "raw_response": response_text
                    }
            
            # Проверка что task_type валиден
            if parsed["task_type"] not in ["classification", "regression"]:
                parsed["task_type"] = "classification"  # fallback
            
            return parsed
            
        except json.JSONDecodeError as e:
            return {
                "error": f"LLM вернул невалидный JSON: {str(e)}",
                "raw_response": response_text if 'response_text' in locals() else "No response",
                "hint": "Попробуйте переформулировать запрос более четко"
            }
        except Exception as e:
            return {
                "error": f"Ошибка при обращении к API: {str(e)}",
                "details": str(type(e).__name__)
            }
    
    def select_best_model(self, models, task_description, data_info=None):
        """
        Выбирает лучшую модель из списка кандидатов
        
        Args:
            models (list): Список названий моделей
            task_description (str): Описание задачи
            data_info (dict, optional): Дополнительная информация о данных
            
        Returns:
            str: Название выбранной модели
        """
        
        data_context = ""
        if data_info:
            data_context = f"\nИнформация о данных: {json.dumps(data_info, ensure_ascii=False)}"
        
        prompt = f"""Выбери ОДНУ лучшую модель для задачи.

Задача: {task_description}{data_context}

Доступные модели: {', '.join(models)}

Верни ТОЛЬКО название модели одним словом, например: RandomForestClassifier

Критерии выбора:
- Для простых линейных задач → Logistic/Linear
- Для средних задач с нелинейностью → RandomForest
- Для сложных задач требующих максимальной точности → GradientBoosting
- Для малых данных → KNeighbors/Ridge
- Для больших данных → LinearRegression/LogisticRegression"""

        try:
            selected = self._call_llm(prompt, max_tokens=50, temperature=0.2)
            selected = selected.strip().split()[0]  # Берем первое слово
            
            # Проверка что модель из списка
            if selected in models:
                return selected
            
            # Если LLM вернул что-то странное, берем первую модель
            print(f"⚠️ LLM вернул неизвестную модель '{selected}', использую {models[0]}")
            return models[0]
            
        except Exception as e:
            print(f"⚠️ Ошибка при выборе модели: {e}")
            return models[0]  # Fallback на первую модель
    
    def suggest_hyperparameters(self, model_name, task_type, data_size=None):
        """
        Предлагает гиперпараметры для модели
        
        Args:
            model_name (str): Название модели
            task_type (str): Тип задачи (classification/regression)
            data_size (int, optional): Размер датасета
            
        Returns:
            dict: Рекомендуемые гиперпараметры в формате scikit-learn
        """
        
        size_context = f"Размер данных: ~{data_size} примеров" if data_size else "Размер данных неизвестен"
        
        prompt = f"""Предложи оптимальные гиперпараметры для модели scikit-learn.

Модель: {model_name}
Задача: {task_type}
{size_context}

Верни JSON с гиперпараметрами, например:
{{
    "n_estimators": 100,
    "max_depth": 10,
    "random_state": 42
}}

Для LinearRegression/LogisticRegression верни просто {{"random_state": 42}}

Верни ТОЛЬКО JSON без текста!"""

        try:
            response_text = self._call_llm(prompt, max_tokens=500, temperature=0.3)
            response_text = re.sub(r'```json\s*', '', response_text)
            response_text = re.sub(r'```\s*', '', response_text)
            response_text = response_text.strip()
            
            params = json.loads(response_text)
            
            # Всегда добавляем random_state для воспроизводимости
            if "random_state" not in params:
                params["random_state"] = 42
            
            return params
            
        except Exception as e:
            print(f"⚠️ Ошибка при получении гиперпараметров: {e}")
            # Возвращаем безопасные дефолтные параметры
            return {"random_state": 42}
    
    def interpret_results(self, metrics, model_name):
        """
        Интерпретирует результаты обучения модели
        
        Args:
            metrics (dict): Метрики модели (accuracy, mse, r2, etc.)
            model_name (str): Название модели
            
        Returns:
            str: Текстовая интерпретация результатов
            
        Example:
            >>> llm.interpret_results({"accuracy": 0.95}, "RandomForest")
            "Модель показывает отличные результаты..."
        """
        
        metrics_str = json.dumps(metrics, ensure_ascii=False, indent=2)
        
        prompt = f"""Проанализируй результаты обучения модели машинного обучения.

Модель: {model_name}
Метрики:
{metrics_str}

Дай краткий анализ (3-4 предложения):
1. Общая оценка качества модели (отлично/хорошо/средне/плохо)
2. Что означают эти метрики простыми словами
3. Есть ли признаки переобучения или недообучения
4. Практические рекомендации по улучшению

Пиши понятным языком, без сложных технических терминов."""

        try:
            interpretation = self._call_llm(prompt, max_tokens=1000, temperature=0.5)
            return interpretation
            
        except Exception as e:
            return f"⚠️ Ошибка при интерпретации результатов: {str(e)}"
    
    def generate_dataset_recommendation(self, task_type, subtask, available_datasets):
        """
        НОВЫЙ: Рекомендует датасет из доступных
        
        Args:
            task_type (str): Тип задачи (classification/regression)
            subtask (str): Подзадача
            available_datasets (list): Список доступных датасетов
            
        Returns:
            str: Рекомендуемый датасет
        """
        
        prompt = f"""Выбери лучший датасет для обучения.

Задача: {task_type} - {subtask}
Доступные датасеты: {', '.join(available_datasets)}

Верни ТОЛЬКО название датасета одним словом, например: load_iris

Критерии:
- Для начинающих → простые датасеты (iris, wine)
- Для медицинских задач → breast_cancer, diabetes
- Для больших задач → digits, california_housing"""

        try:
            selected = self._call_llm(prompt, max_tokens=50, temperature=0.2)
            selected = selected.strip().split()[0]
            
            if selected in available_datasets:
                return selected
            
            return available_datasets[0]
            
        except Exception as e:
            print(f"⚠️ Ошибка выбора датасета: {e}")
            return available_datasets[0]


# Тестирование (запускается только при прямом вызове файла)
if __name__ == "__main__":
    print("🧪 Тестирование LLMOptimizer...\n")
    
    try:
        llm = LLMOptimizer()
        
        # Тест 1: Парсинг задачи классификации
        print("📝 Тест 1: Парсинг задачи классификации")
        result1 = llm.parse_task("Хочу классифицировать цветы ириса по размеру лепестков")
        print(json.dumps(result1, ensure_ascii=False, indent=2))
        
        print("\n" + "="*50 + "\n")
        
        # Тест 2: Парсинг задачи регрессии
        print("📝 Тест 2: Парсинг задачи регрессии")
        result2 = llm.parse_task("Нужно предсказывать цены на квартиры по площади")
        print(json.dumps(result2, ensure_ascii=False, indent=2))
        
        print("\n" + "="*50 + "\n")
        
        # Тест 3: Выбор модели
        print("📝 Тест 3: Выбор лучшей модели")
        models = ["LinearRegression", "Ridge", "RandomForestRegressor"]
        best = llm.select_best_model(models, "предсказание цен на недвижимость")
        print(f"Лучшая модель: {best}")
        
        print("\n✅ Все тесты пройдены!")
        
    except ValueError as e:
        print(f"❌ Ошибка: {e}")
        print("\n💡 Создайте файл .env в корне проекта:")
        print("OPENAI_API_KEY=sk-ваш-ключ-здесь")
    except Exception as e:
        print(f"❌ Неожиданная ошибка: {e}")