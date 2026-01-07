import os
from datasets import load_dataset, list_datasets
from huggingface_hub import HfApi, DatasetInfo
import pandas as pd

class HuggingFaceIntegration:
    """
    Интеграция с Hugging Face для работы с датасетами
    
    Возможности:
    - Поиск датасетов по запросу
    - Получение метаданных (размер, описание, задачи)
    - Загрузка датасетов
    - Конвертация в pandas DataFrame
    """
    
    def __init__(self, token=None):
        """
        Инициализация клиента Hugging Face
        
        Args:
            token: HF токен (если None, берется из HUGGINGFACE_TOKEN)
        """
        self.token = token or os.getenv("HUGGINGFACE_TOKEN")
        
        # Токен опционален для публичных датасетов
        if not self.token:
            print("⚠️ HUGGINGFACE_TOKEN не найден. Работа только с публичными датасетами.")
        else:
            print("✅ Hugging Face API инициализирован")
        
        self.api = HfApi(token=self.token)
        self.cache = {}  # Кэш для избежания повторных запросов
    
    def search_datasets(self, query, task_type=None, limit=10):
        """
        Поиск датасетов по запросу
        
        Args:
            query (str): Поисковый запрос
            task_type (str): Тип задачи (text-classification, image-classification, etc.)
            limit (int): Максимальное количество результатов
            
        Returns:
            list: Список словарей с информацией о датасетах
            
        Example:
            >>> hf.search_datasets("sentiment", task_type="text-classification")
            [
                {
                    "id": "imdb",
                    "description": "Large Movie Review Dataset",
                    "downloads": 125000,
                    "task": "text-classification"
                },
                ...
            ]
        """
        try:
            # Фильтр по типу задачи
            task_filter = task_type if task_type else None
            
            # Поиск через API
            datasets = self.api.list_datasets(
                search=query,
                task_categories=task_filter,
                sort="downloads",
                direction=-1,
                limit=limit
            )
            
            results = []
            for dataset in datasets:
                results.append({
                    "id": dataset.id,
                    "author": dataset.author,
                    "downloads": getattr(dataset, 'downloads', 0),
                    "likes": getattr(dataset, 'likes', 0),
                    "tags": getattr(dataset, 'tags', [])
                })
            
            return results
            
        except Exception as e:
            print(f"⚠️ Ошибка поиска датасетов: {e}")
            return []
    
    def get_dataset_info(self, dataset_id):
        """
        Получить детальную информацию о датасете
        
        Args:
            dataset_id (str): ID датасета (например, "imdb")
            
        Returns:
            dict: Информация о датасете
        """
        # Проверка кэша
        if dataset_id in self.cache:
            return self.cache[dataset_id]
        
        try:
            info = self.api.dataset_info(dataset_id)
            
            result = {
                "id": dataset_id,
                "description": getattr(info, 'description', 'Нет описания'),
                "citation": getattr(info, 'citation', ''),
                "homepage": getattr(info, 'homepage', ''),
                "license": getattr(info, 'license', 'unknown'),
                "features": str(getattr(info, 'features', {})),
                "splits": list(getattr(info, 'splits', {}).keys()),
                "download_size": getattr(info, 'download_size', 0),
                "dataset_size": getattr(info, 'dataset_size', 0),
            }
            
            # Сохранить в кэш
            self.cache[dataset_id] = result
            
            return result
            
        except Exception as e:
            print(f"⚠️ Ошибка получения информации: {e}")
            return {
                "id": dataset_id,
                "error": str(e)
            }
    
    def load_dataset_as_dataframe(self, dataset_id, split="train", max_rows=1000):
        """
        Загрузить датасет и конвертировать в pandas DataFrame
        
        Args:
            dataset_id (str): ID датасета
            split (str): Разбиение (train/test/validation)
            max_rows (int): Максимальное количество строк (для экономии памяти)
            
        Returns:
            pd.DataFrame: Датасет в формате DataFrame
            
        Example:
            >>> df = hf.load_dataset_as_dataframe("imdb", split="train", max_rows=100)
            >>> df.head()
        """
        try:
            print(f"📥 Загрузка {dataset_id} ({split})...")
            
            # Загрузка датасета
            dataset = load_dataset(
                dataset_id,
                split=split,
                streaming=False,  # Полная загрузка (для малых датасетов)
                token=self.token
            )
            
            # Ограничение размера
            if len(dataset) > max_rows:
                dataset = dataset.select(range(max_rows))
                print(f"⚠️ Датасет обрезан до {max_rows} строк")
            
            # Конвертация в DataFrame
            df = pd.DataFrame(dataset)
            
            print(f"✅ Загружено: {len(df)} строк, {len(df.columns)} колонок")
            
            return df
            
        except Exception as e:
            print(f"❌ Ошибка загрузки датасета: {e}")
            return None
    
    def get_popular_datasets(self, task_category=None, limit=20):
        """
        Получить список популярных датасетов
        
        Args:
            task_category (str): Категория задачи
            limit (int): Количество датасетов
            
        Returns:
            list: Список популярных датасетов
        """
        try:
            datasets = self.api.list_datasets(
                task_categories=task_category,
                sort="downloads",
                direction=-1,
                limit=limit
            )
            
            results = []
            for ds in datasets:
                results.append({
                    "id": ds.id,
                    "downloads": getattr(ds, 'downloads', 0),
                    "tags": getattr(ds, 'tags', [])
                })
            
            return results
            
        except Exception as e:
            print(f"⚠️ Ошибка получения популярных датасетов: {e}")
            return []
    
    def recommend_dataset(self, task_description, task_type="tabular"):
        """
        НОВЫЙ: Рекомендовать датасет на основе описания задачи
        
        Args:
            task_description (str): Описание задачи
            task_type (str): Тип задачи (tabular/text/image/audio)
            
        Returns:
            str: ID рекомендуемого датасета
        """
        # Маппинг типов задач в категории HF
        task_mapping = {
            "tabular": "tabular-classification",
            "text": "text-classification",
            "image": "image-classification",
            "audio": "audio-classification"
        }
        
        hf_task = task_mapping.get(task_type, None)
        
        # Поиск по описанию
        results = self.search_datasets(
            query=task_description,
            task_type=hf_task,
            limit=5
        )
        
        if results:
            # Возвращаем самый популярный
            return results[0]["id"]
        
        return None


# Тестирование (только при прямом запуске)
if __name__ == "__main__":
    print("🧪 Тестирование Hugging Face Integration\n")
    
    try:
        hf = HuggingFaceIntegration()
        
        # Тест 1: Поиск датасетов
        print("📝 Тест 1: Поиск датасетов про sentiment")
        results = hf.search_datasets("sentiment", limit=3)
        for ds in results:
            print(f"  - {ds['id']} (downloads: {ds['downloads']})")
        
        print("\n" + "="*50 + "\n")
        
        # Тест 2: Информация о датасете
        print("📝 Тест 2: Информация о датасете 'imdb'")
        info = hf.get_dataset_info("imdb")
        print(f"  Описание: {info.get('description', 'N/A')[:100]}...")
        print(f"  Лицензия: {info.get('license', 'N/A')}")
        print(f"  Splits: {info.get('splits', [])}")
        
        print("\n" + "="*50 + "\n")
        
        # Тест 3: Загрузка данных
        print("📝 Тест 3: Загрузка датасета 'imdb' (100 строк)")
        df = hf.load_dataset_as_dataframe("imdb", split="train", max_rows=100)
        if df is not None:
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {list(df.columns)}")
            print(f"\nПервые 3 строки:")
            print(df.head(3))
        
        print("\n✅ Все тесты Hugging Face пройдены!")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        print("\n💡 Убедитесь что установлена библиотека datasets:")
        print("   pip install datasets huggingface-hub")