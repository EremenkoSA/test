"""
Модуль для оценки качества перевода с эталоном (Reference-based Evaluation)

Используемые метрики:
- COMET (Crosslingual Optimized Metric for Evaluation of Translation) - нейросетевая метрика,
  учитывающая семантику и контекст (заменяет BLEU/ROUGE/BERTScore)
"""

from typing import List, Tuple, Dict
import evaluate


class ReferenceBasedEvaluator:
    """Оценка качества перевода с использованием эталонных переводов на основе COMET"""

    def __init__(self):
        print("Loading COMET metric (wmt22-comet-da)...")
        try:
            # Загружаем предобученную модель COMET
            self.comet_metric = evaluate.load("comet", "Unbabel/wmt22-comet-da")
            print("COMET metric loaded successfully.")
        except Exception as e:
            print(f"Error loading COMET: {e}")
            raise e

    def calculate_comet(
            self,
            sources: List[str],
            candidates: List[str],
            references: List[str]
    ) -> Dict[str, float]:
        """
        Расчет метрики COMET

        COMET использует предобученную языковую модель для оценки качества перевода,
        учитывая исходный текст, гипотезу и референс. Это дает более точную оценку,
        чем n-gram метрики типа BLEU.

        Args:
            sources: Список исходных текстов
            candidates: Список переведенных текстов (гипотезы)
            references: Список эталонных переводов

        Returns:
            Словарь с результатами COMET (mean_score, scores list)
        """
        if len(sources) != len(candidates) or len(sources) != len(references):
            raise ValueError("Sources, candidates, and references must have the same length")

        # COMET требует формат данных: список словарей с ключами 'src', 'mt', 'ref'
        # Данные передаются напрямую в compute через именованные аргументы

        print(f"Calculating COMET for {len(sources)} samples...")

        # Формируем данные в формате, требуемом COMET
        data = {
            "src": sources,
            "mt": candidates,
            "ref": references
        }

        # Вычисляем метрику через compute
        # Для evaluate.load интерфейс использует compute с отдельными аргументами:
        # sources, predictions, references
        try:
            results = self.comet_metric.compute(
                sources=data["src"],
                predictions=data["mt"],
                references=data["ref"],
                progress_bar=False
            )

            # Обработка различных форматов вывода COMET
            # В новых версиях может быть 'mean_score' вместо 'system_score'
            mean_score = results.get("system_score") or results.get("mean_score")

            # Если нет готового среднего, считаем его из списка оценок
            if mean_score is None and "scores" in results:
                scores_list = results["scores"]
                if scores_list:
                    mean_score = sum(scores_list) / len(scores_list)

            # Fallback к 0, если ничего не нашли
            if mean_score is None:
                mean_score = 0.0

            scores_list = results.get("scores", [0.0] * len(sources))

            return {
                "comet_mean": float(mean_score),
                "comet_scores": [float(s) for s in scores_list]
            }
        except Exception as e:
            print(f"Error calculating COMET: {e}")
            # Fallback to 0 if calculation fails
            return {
                "comet_mean": 0.0,
                "comet_scores": [0.0] * len(sources)
            }

    def evaluate_all(
            self,
            sources: List[str],
            candidates: List[str],
            references: List[str]
    ) -> Dict[str, Dict]:
        """
        Полная оценка с использованием COMET

        Args:
            sources: Список исходных текстов
            candidates: Список переведенных текстов
            references: Список эталонных переводов

        Returns:
            Полный словарь с метрикой COMET
        """
        results = {}

        print("Calculating COMET (primary reference metric)...")
        results["comet"] = self.calculate_comet(sources, candidates, references)

        return results


def parse_dataset_line(line: str) -> Tuple[str, str]:
    """
    Парсинг строки датасета формата: исходный текст ||| эталонный перевод

    Args:
        line: Строка из файла датасета

    Returns:
        Кортеж (исходный текст, эталонный перевод)
    """
    parts = line.strip().split(" ||| ")
    if len(parts) >= 2:
        return parts[0].strip(), parts[1].strip()
    return None, None


def load_dataset(filepath: str, max_lines: int = None) -> List[Tuple[str, str]]:
    """
    Загрузка датасета с параллельными текстами

    Args:
        filepath: Путь к файлу датасета
        max_lines: Максимальное количество строк для загрузки (None = все)

    Returns:
        Список кортежей (исходный текст, эталон)
    """
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                break
            source, reference = parse_dataset_line(line)
            if source and reference:
                data.append((source, reference))
    return data


if __name__ == "__main__":
    # Тестовый пример использования
    evaluator = ReferenceBasedEvaluator()

    # Пример данных
    sources = [
        "Кошка сидела на коврике.",
        "Это тестовый перевод."
    ]
    candidates = [
        "The cat sat on the mat.",
        "This is a test translation."
    ]
    references = [
        "The cat sat on the mat.",
        "This is a sample translation."
    ]

    results = evaluator.evaluate_all(sources, candidates, references)

    print("\n=== Results ===")
    print(f"COMET Mean Score: {results['comet']['comet_mean']:.4f}")
    print(f"Individual Scores: {results['comet']['comet_scores']}")