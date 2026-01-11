# integrations/kaggle_api.py

import os
import json
from pathlib import Path
from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd

# Загрузить переменные из .env
load_dotenv()

class KaggleIntegration:
    """
    Интеграция с Kaggle для работы с датасетами
    
    Возможности:
    - Поиск датасетов по запросу
    - Получение метаданных
    - Скачивание датасетов
    - Автоматическое распаковывание
    """
    
    def __init__(self, username=None, key=None):
        """
        Инициализация Kaggle API
        
        Args:
            username: Kaggle username (если None, берется из .env)
            key: Kaggle API key (если None, берется из .env)
            
        Примечание:
            Kaggle также может читать credentials из ~/.kaggle/kaggle.json
        """
        self.username = username or os.getenv("KAGGLE_USERNAME")
        self.key = key or os.getenv("KAGGLE_KEY")
        
        # Настройка credentials
        if self.username and self.key:
            os.environ["KAGGLE_USERNAME"] = self.username
            os.environ["KAGGLE_KEY"] = self.key
            print("✅ Kaggle API инициализирован через .env")
        else:
            # Попытка использовать ~/.kaggle/kaggle.json
            kaggle_config = Path.home() / ".kaggle" / "kaggle.json"
            if kaggle_config.exists():
                print("✅ Kaggle API инициализирован через ~/.kaggle/kaggle.json")
            else:
                raise ValueError(
                    "❌ Kaggle credentials не найдены!\n"
                    "Добавьте в .env:\n"
                    "  KAGGLE_USERNAME=ваш_username\n"
                    "  KAGGLE_KEY=ваш_ключ\n"
                    "Или создайте ~/.kaggle/kaggle.json"
                )
        
        self.api = KaggleApi()
        self.api.authenticate()
        
        # Папка для кэша датасетов
        self.cache_dir = Path("datasets_cache")
        self.cache_dir.mkdir(exist_ok=True)
    
    def search_datasets(self, query, sort_by="hotness", limit=10):
        """
        Поиск датасетов по запросу
        
        Args:
            query (str): Поисковый запрос
            sort_by (str): Сортировка (hotness/votes/updated/active)
            limit (int): Максимальное количество результатов
            
        Returns:
            list: Список датасетов
            
        Example:
            >>> kg.search_datasets("titanic", limit=5)
            [
                {
                    "ref": "username/dataset-name",
                    "title": "Titanic Dataset",
                    "size": 12345,
                    "votes": 150,
                    ...
                }
            ]
        """
        try:
            print(f"🔍 Поиск датасетов Kaggle: '{query}'...")
            
            datasets = self.api.dataset_list(
                search=query,
                sort_by=sort_by,
                page=1,
                max_size=limit
            )
            
            results = []
            for ds in datasets[:limit]:
                results.append({
                    "ref": ds.ref,  # username/dataset-name
                    "title": ds.title,
                    "size": ds.size,
                    "votes": ds.voteCount,
                    "downloads": ds.downloadCount,
                    "last_updated": str(ds.lastUpdated),
                    "url": f"https://www.kaggle.com/datasets/{ds.ref}"
                })
            
            print(f"✅ Найдено {len(results)} датасетов")
            return results
            
        except Exception as e:
            print(f"❌ Ошибка поиска: {e}")
            return []
    
    def get_dataset_metadata(self, dataset_ref):
        """
        Получить метаданные датасета
        
        Args:
            dataset_ref (str): Ссылка на датасет (username/dataset-name)
            
        Returns:
            dict: Метаданные датасета
        """
        try:
            # Разделить ref на owner и dataset
            owner, dataset_name = dataset_ref.split("/")
            
            metadata = self.api.dataset_metadata(owner, dataset_name)
            
            return {
                "ref": dataset_ref,
                "id": metadata.id,
                "title": metadata.title,
                "description": metadata.description,
                "size": metadata.totalBytes,
                "license": metadata.licenseName,
                "files": [f.name for f in metadata.datasetFiles] if metadata.datasetFiles else []
            }
            
        except Exception as e:
            print(f"⚠️ Ошибка получения метаданных: {e}")
            return {"ref": dataset_ref, "error": str(e)}
    
    def download_dataset(self, dataset_ref, path=None, unzip=True):
        """
        Скачать датасет
        
        Args:
            dataset_ref (str): Ссылка на датасет (username/dataset-name)
            path (str): Путь для сохранения (если None, используется cache_dir)
            unzip (bool): Распаковывать ли архив
            
        Returns:
            str: Путь к скачанным файлам
            
        Example:
            >>> kg.download_dataset("username/titanic")
            "datasets_cache/titanic/"
        """
        try:
            # Определить путь
            if path is None:
                dataset_name = dataset_ref.split("/")[1]
                path = self.cache_dir / dataset_name
            else:
                path = Path(path)
            
            path.mkdir(parents=True, exist_ok=True)
            
            print(f"📥 Скачивание {dataset_ref}...")
            
            # Скачать датасет
            self.api.dataset_download_files(
                dataset=dataset_ref,
                path=str(path),
                unzip=unzip,
                quiet=False
            )
            
            print(f"✅ Датасет скачан в {path}")
            
            return str(path)
            
        except Exception as e:
            print(f"❌ Ошибка скачивания: {e}")
            return None
    
    def load_dataset_as_dataframe(self, dataset_ref, file_name=None, max_rows=1000):
        """
        Скачать и загрузить датасет как pandas DataFrame
        
        Args:
            dataset_ref (str): Ссылка на датасет
            file_name (str): Имя CSV файла (если None, берется первый CSV)
            max_rows (int): Максимальное количество строк
            
        Returns:
            pd.DataFrame: Датасет
            
        Example:
            >>> df = kg.load_dataset_as_dataframe("username/titanic")
            >>> df.head()
        """
        try:
            # Скачать датасет
            dataset_path = self.download_dataset(dataset_ref)
            if not dataset_path:
                return None
            
            # Найти CSV файлы
            csv_files = list(Path(dataset_path).glob("*.csv"))
            
            if not csv_files:
                print("⚠️ CSV файлы не найдены в датасете")
                return None
            
            # Выбрать файл
            if file_name:
                csv_file = Path(dataset_path) / file_name
            else:
                csv_file = csv_files[0]  # Первый найденный CSV
            
            print(f"📊 Загрузка {csv_file.name}...")
            
            # Загрузить CSV
            df = pd.read_csv(csv_file, nrows=max_rows)
            
            print(f"✅ Загружено: {len(df)} строк, {len(df.columns)} колонок")
            
            return df
            
        except Exception as e:
            print(f"❌ Ошибка загрузки: {e}")
            return None
    
    def get_popular_datasets(self, category=None, limit=20):
        """
        Получить список популярных датасетов
        
        Args:
            category (str): Категория (если None, все категории)
            limit (int): Количество датасетов
            
        Returns:
            list: Список популярных датасетов
        """
        try:
            datasets = self.api.dataset_list(
                sort_by="hotness",
                page=1,
                max_size=limit
            )
            
            results = []
            for ds in datasets[:limit]:
                results.append({
                    "ref": ds.ref,
                    "title": ds.title,
                    "votes": ds.voteCount,
                    "downloads": ds.downloadCount
                })
            
            return results
            
        except Exception as e:
            print(f"⚠️ Ошибка: {e}")
            return []
    
    def list_files_in_dataset(self, dataset_ref):
        """
        Получить список файлов в датасете (без скачивания)
        
        Args:
            dataset_ref (str): Ссылка на датасет
            
        Returns:
            list: Список имен файлов
        """
        try:
            owner, dataset_name = dataset_ref.split("/")
            
            files = self.api.dataset_list_files(owner, dataset_name)
            
            return [f.name for f in files.files]
            
        except Exception as e:
            print(f"⚠️ Ошибка: {e}")
            return []


# Тестирование (только при прямом запуске)
if __name__ == "__main__":
    print("🧪 Тестирование Kaggle Integration\n")
    
    try:
        kg = KaggleIntegration()
        
        # Тест 1: Поиск датасетов
        print("📝 Тест 1: Поиск датасетов 'titanic'")
        results = kg.search_datasets("titanic", limit=3)
        for ds in results:
            print(f"  - {ds['title']} ({ds['ref']})")
            print(f"    Votes: {ds['votes']}, Downloads: {ds['downloads']}")
        
        print("\n" + "="*50 + "\n")
        
        # Тест 2: Метаданные датасета
        if results:
            dataset_ref = results[0]["ref"]
            print(f"📝 Тест 2: Метаданные '{dataset_ref}'")
            metadata = kg.get_dataset_metadata(dataset_ref)
            print(f"  Описание: {metadata.get('description', 'N/A')[:100]}...")
            print(f"  Размер: {metadata.get('size', 0) / 1024:.2f} KB")
            print(f"  Файлы: {metadata.get('files', [])}")
        
        print("\n" + "="*50 + "\n")
        
        # Тест 3: Список файлов
        if results:
            print(f"📝 Тест 3: Список файлов в '{dataset_ref}'")
            files = kg.list_files_in_dataset(dataset_ref)
            print(f"  Файлов: {len(files)}")
            for f in files[:5]:
                print(f"    - {f}")
        
        print("\n✅ Все тесты Kaggle пройдены!")
        print("\n💡 Для полного теста скачивания раскомментируйте следующий код:")
        print("# df = kg.load_dataset_as_dataframe(dataset_ref, max_rows=100)")
        
    except ValueError as e:
        print(f"❌ {e}")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        print("\n💡 Убедитесь что:")
        print("   1. Установлена библиотека: pip install kaggle")
        print("   2. Настроены credentials (см. документацию в коде)")