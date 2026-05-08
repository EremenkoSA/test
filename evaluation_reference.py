"""
Модуль для оценки качества перевода с эталоном (Reference-based Evaluation)

Используемые метрики:
- BLEU (Bilingual Evaluation Understudy)
- ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
- BERTScore (Semantic similarity using BERT embeddings)
"""

from typing import List, Tuple, Dict
import sacrebleu
from rouge_score import rouge_scorer
from bert_score import score as bert_score


class ReferenceBasedEvaluator:
    """Оценка качества перевода с использованием эталонных переводов"""

    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=True
        )

    def calculate_bleu(
            self,
            candidates: List[str],
            references: List[str]
    ) -> Dict[str, float]:
        """
        Расчет метрики BLEU

        Args:
            candidates: Список переведенных текстов (гипотезы)
            references: Список эталонных переводов

        Returns:
            Словарь с результатами BLEU (score, counts, totals)
        """
        # sacrebleu ожидает список списков для референсов
        references_list = [[ref] for ref in references]

        bleu_result = sacrebleu.corpus_bleu(candidates, references_list)

        return {
            "bleu_score": bleu_result.score,
            "bleu_counts": bleu_result.counts,
            "bleu_totals": bleu_result.totals,
            "bleu_precisions": bleu_result.precisions
        }

    def calculate_rouge(
            self,
            candidates: List[str],
            references: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Расчет метрик ROUGE-1, ROUGE-2, ROUGE-L

        Args:
            candidates: Список переведенных текстов
            references: Список эталонных переводов

        Returns:
            Словарь с результатами по каждой метрике (precision, recall, fmeasure)
        """
        results = {"rouge1": [], "rouge2": [], "rougeL": []}

        for candidate, reference in zip(candidates, references):
            scores = self.rouge_scorer.score(reference, candidate)
            for metric in results.keys():
                results[metric].append({
                    "precision": scores[metric].precision,
                    "recall": scores[metric].recall,
                    "fmeasure": scores[metric].fmeasure
                })

        # Усредняем результаты
        averaged_results = {}
        for metric, scores_list in results.items():
            if scores_list:
                averaged_results[metric] = {
                    "precision": sum(s["precision"] for s in scores_list) / len(scores_list),
                    "recall": sum(s["recall"] for s in scores_list) / len(scores_list),
                    "fmeasure": sum(s["fmeasure"] for s in scores_list) / len(scores_list)
                }

        return averaged_results

    def calculate_bertscore(
            self,
            candidates: List[str],
            references: List[str],
            lang: str = "en"
    ) -> Dict[str, float]:
        """
        Расчет метрики BERTScore (семантическое сходство)

        Args:
            candidates: Список переведенных текстов
            references: Список эталонных переводов
            lang: Язык текстов (для выбора правильной модели)

        Returns:
            Словарь с precision, recall, F1
        """
        # map language to bert-score language code
        lang_code = self._get_bertscore_lang_code(lang)

        P, R, F1 = bert_score(candidates, references, lang=lang_code, verbose=False)

        return {
            "bert_precision": P.mean().item(),
            "bert_recall": R.mean().item(),
            "bert_f1": F1.mean().item()
        }

    def _get_bertscore_lang_code(self, lang: str) -> str:
        """Преобразование названия языка в код для BERTScore"""
        lang_mapping = {
            "english": "en",
            "russian": "ru",
            "german": "de",
            "french": "fr",
            "spanish": "es",
            "en": "en",
            "ru": "ru",
            "de": "de",
            "fr": "fr",
            "es": "es"
        }
        return lang_mapping.get(lang.lower(), "en")

    def evaluate_all(
            self,
            candidates: List[str],
            references: List[str],
            target_lang: str = "en"
    ) -> Dict[str, Dict]:
        """
        Полная оценка всеми доступными метриками

        Args:
            candidates: Список переведенных текстов
            references: Список эталонных переводов
            target_lang: Язык перевода

        Returns:
            Полный словарь со всеми метриками
        """
        results = {}

        print("Calculating BLEU...")
        results["bleu"] = self.calculate_bleu(candidates, references)

        print("Calculating ROUGE...")
        results["rouge"] = self.calculate_rouge(candidates, references)

        print("Calculating BERTScore...")
        results["bertscore"] = self.calculate_bertscore(candidates, references, target_lang)

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
    candidates = [
        "The cat sat on the mat.",
        "This is a test translation."
    ]
    references = [
        "The cat sat on the mat.",
        "This is a sample translation."
    ]

    results = evaluator.evaluate_all(candidates, references)

    print("\n=== Results ===")
    for metric, scores in results.items():
        print(f"\n{metric.upper()}:")
        for score_name, value in scores.items():
            print(f"  {score_name}: {value}")