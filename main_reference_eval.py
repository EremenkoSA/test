"""
Основной скрипт для оценки перевода с эталоном

Использует LM Studio для перевода и оценивает качество с помощью:
- BLEU
- ROUGE
- BERTScore
"""

import json
import time
from typing import List, Tuple
from lmstudio_client import LMStudioClient
from evaluation_reference import ReferenceBasedEvaluator, load_dataset


def run_reference_evaluation(
        dataset_path: str = "datasets/rus-eng-part1-text.txt",
        num_samples: int = 10,
        source_lang: str = "Russian",
        target_lang: str = "English",
        model: str = "gigachat3.1-10b-a1.8b"
):
    """
    Запуск оценки перевода с эталоном

    Args:
        dataset_path: Путь к файлу датасета
        num_samples: Количество образцов для тестирования
        source_lang: Язык исходных текстов
        target_lang: Язык перевода
        model: Модель для использования
    """
    print("=" * 60)
    print("ОЦЕНКА КАЧЕСТВА ПЕРЕВОДА С ЭТАЛОНОМ")
    print("=" * 60)

    # Инициализация клиентов
    llm_client = LMStudioClient()
    evaluator = ReferenceBasedEvaluator()

    # Загрузка датасета
    print(f"\nЗагрузка датасета из {dataset_path}...")
    data = load_dataset(dataset_path, max_lines=num_samples)
    print(f"Загружено {len(data)} пар текст-эталон")

    if not data:
        print("Ошибка: не удалось загрузить данные")
        return

    # Перевод текстов с помощью нейросети
    print(f"\nВыполнение перевода ({len(data)} текстов)...")
    candidates = []
    references = []
    sources = []

    for i, (source_text, reference) in enumerate(data):
        print(f"  [{i+1}/{len(data)}] Перевод: {source_text[:50]}...")

        # Получаем перевод от модели
        translation = llm_client.translate(
            text=source_text,
            source_lang=source_lang,
            target_lang=target_lang,
            model=model
        )

        if translation:
            candidates.append(translation)
            references.append(reference)
            sources.append(source_text)
            print(f"       -> {translation[:50]}...")
        else:
            print(f"       [ERROR] Не удалось получить перевод")

        # Небольшая задержка между запросами
        time.sleep(0.5)

    if not candidates:
        print("\nОшибка: не удалось получить ни одного перевода")
        return

    print(f"\nУспешно переведено {len(candidates)} текстов")

    # Оценка качества
    print("\n" + "=" * 60)
    print("ВЫЧИСЛЕНИЕ МЕТРИК КАЧЕСТВА")
    print("=" * 60)

    results = evaluator.evaluate_all(candidates, references, target_lang)

    # Вывод результатов
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ ОЦЕНКИ")
    print("=" * 60)

    print(f"\n{'Метрика':<25} {'Значение':<15}")
    print("-" * 40)

    # chrF++ (основная метрика)
    chrf_pp = results.get("chrf_pp", {})
    print(f"chrF++ (основная):       {chrf_pp.get('chrf_pp_score', 'N/A'):.4f}")

    # BLEU (для сравнения)
    bleu = results.get("bleu", {})
    print(f"BLEU Score (сравнение):  {bleu.get('bleu_score', 'N/A'):.4f}")

    # ROUGE
    rouge = results.get("rouge", {})
    for metric in ["rouge1", "rouge2", "rougeL"]:
        if metric in rouge:
            f1 = rouge[metric].get("fmeasure", 0)
            print(f"{metric.upper()} F1:                 {f1:.4f}")

    # BERTScore
    bertscore = results.get("bertscore", {})
    print(f"BERTScore Precision:     {bertscore.get('bert_precision', 'N/A'):.4f}")
    print(f"BERTScore Recall:        {bertscore.get('bert_recall', 'N/A'):.4f}")
    print(f"BERTScore F1:            {bertscore.get('bert_f1', 'N/A'):.4f}")

    # Сохранение результатов в JSON
    output_data = {
        "config": {
            "model": model,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "num_samples": num_samples
        },
        "metrics": results,
        "samples": [
            {
                "source": src,
                "translation": trans,
                "reference": ref
            }
            for src, trans, ref in zip(sources, candidates, references)
        ]
    }

    output_file = "results_reference_based.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\nРезультаты сохранены в {output_file}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Оценка качества перевода с эталоном"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="datasets/rus-eng-part1-text.txt",
        help="Путь к файлу датасета"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10,
        help="Количество образцов для тестирования"
    )
    parser.add_argument(
        "--source-lang",
        type=str,
        default="Russian",
        help="Язык исходных текстов"
    )
    parser.add_argument(
        "--target-lang",
        type=str,
        default="English",
        help="Язык перевода"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gigachat3.1-10b-a1.8b",
        help="Название модели"
    )

    args = parser.parse_args()

    run_reference_evaluation(
        dataset_path=args.dataset,
        num_samples=args.samples,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        model=args.model
    )