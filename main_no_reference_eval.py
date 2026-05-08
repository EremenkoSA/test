#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Оценка качества перевода БЕЗ эталона (No-Reference Evaluation)
Метод: Обратный перевод (Back-Translation)

Логика:
1. Текст (Source) -> Перевод (Target)
2. Перевод (Target) -> Обратный перевод (Back-Translated Source)
3. Сравнение: Source vs Back-Translated Source

Чем ближе обратный перевод к оригиналу, тем выше качество прямого перевода.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('no_reference_debug.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

try:
    import requests
    import sacrebleu
    from rouge_score import rouge_scorer
    from bert_score import score as bert_score
except ImportError as e:
    logger.error(f"Ошибка импорта библиотек: {e}")
    logger.info("Установите зависимости: pip install requests sacrebleu rouge-score bert-score")
    sys.exit(1)


class LMStudioClient:
    """Клиент для взаимодействия с локальным сервером LM Studio."""

    def __init__(self, base_url: str = "http://localhost:1234/api/v1/chat", model: str = "gigachat3.1-10b-a1.8b"):
        self.base_url = base_url
        self.model = model
        logger.info(f"Инициализация клиента для оценки без эталона. API endpoint: {base_url}")

    def translate(self, text: str, source_lang: str, target_lang: str) -> str | None:
        """Выполняет перевод текста."""
        system_instruction = (
            f"You are a professional translator. Translate the following text from {source_lang} to {target_lang}. "
            "Return ONLY the translated text, without any explanations."
        )

        payload = {
            "model": self.model,
            "system_prompt": system_instruction,
            "input": text
        }

        logger.debug(f"Запрос перевода: {source_lang} -> {target_lang}")
        logger.debug(f"Текст (первые 50 символов): {text[:50]}...")

        try:
            response = requests.post(self.base_url, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()

            # Парсинг ответа (формат Gigachat/LM Studio)
            translation = None
            if "output" in data and isinstance(data["output"], list) and len(data["output"]) > 0:
                item = data["output"][0]
                if isinstance(item, dict) and "content" in item:
                    translation = item["content"]

            if not translation:
                logger.warning(f"Не удалось извлечь перевод из ответа: {data}")
                return None

            return translation.strip()

        except requests.exceptions.RequestException as e:
            logger.error(f"Ошибка сети при запросе: {e}")
            return None
        except Exception as e:
            logger.error(f"Неожиданная ошибка: {e}")
            return None

    def back_translate(self, text: str, source_lang: str, original_lang: str) -> str | None:
        """Выполняет обратный перевод для проверки качества."""
        # Если исходный язык был 'auto', предполагаем английский для обратного пути
        effective_source = original_lang if original_lang != 'auto' else 'en'

        system_instruction = (
            f"You are a professional translator. Translate the following text from {source_lang} to {effective_source}. "
            "Return ONLY the translated text, without any explanations."
        )

        payload = {
            "model": self.model,
            "system_prompt": system_instruction,
            "input": text
        }

        logger.debug(f"Запрос обратного перевода: {source_lang} -> {effective_source}")
        logger.debug(f"Текст (первые 50 символов): {text[:50]}...")

        try:
            response = requests.post(self.base_url, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()

            translation = None
            if "output" in data and isinstance(data["output"], list) and len(data["output"]) > 0:
                item = data["output"][0]
                if isinstance(item, dict) and "content" in item:
                    translation = item["content"]

            if not translation:
                logger.warning(f"Не удалось извлечь обратный перевод из ответа: {data}")
                return None

            return translation.strip()

        except requests.exceptions.RequestException as e:
            logger.error(f"Ошибка сети при обратном переводе: {e}")
            return None
        except Exception as e:
            logger.error(f"Неожиданная ошибка при обратном переводе: {e}")
            return None


def load_dataset(file_path: str, num_samples: int) -> List[str]:
    """Загружает тексты из файла (по одному тексту на строку или разделенные табуляцией)."""
    texts = []
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {file_path}")

    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Если в строке есть табуляция, берем первую часть (исходный текст)
        # Формат файла может быть просто текстом или парами "текст\tэталон"
        parts = line.split('\t')
        texts.append(parts[0])

        if len(texts) >= num_samples:
            break

    return texts


def calculate_metrics(sources: List[str], back_translations: List[str]) -> Dict[str, Any]:
    """Вычисляет метрики между оригиналом и обратным переводом."""
    results = {}

    # 1. BLEU
    logger.info("Calculating BLEU (Source vs Back-Translation)...")
    # SacreBLEU ожидает список списков для референсов
    bleu_result = sacrebleu.corpus_bleu(back_translations, [[s] for s in sources])
    results['bleu'] = {
        'score': bleu_result.score,
        'counts': bleu_result.counts,
        'totals': bleu_result.totals,
        'precisions': bleu_result.precisions
    }

    # 2. ROUGE
    logger.info("Calculating ROUGE (Source vs Back-Translation)...")
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
    rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}

    for src, back in zip(sources, back_translations):
        scores = scorer.score(src, back)
        for key in rouge_scores:
            rouge_scores[key].append(scores[key].fmeasure)

    results['rouge'] = {
        key: sum(vals) / len(vals) if vals else 0.0
        for key, vals in rouge_scores.items()
    }

    # 3. BERTScore
    logger.info("Calculating BERTScore (Source vs Back-Translation)...")
    # BERTScore лучше работает, если языки совпадают (RU vs RU)
    P, R, F1 = bert_score(back_translations, sources, lang='ru', verbose=True)
    results['bertscore'] = {
        'precision': P.mean().item(),
        'recall': R.mean().item(),
        'f1': F1.mean().item()
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Оценка перевода без эталона (Back-Translation)")
    parser.add_argument("--dataset", type=str, required=True, help="Путь к файлу с текстами")
    parser.add_argument("--samples", type=int, default=5, help="Количество текстов для обработки")
    parser.add_argument("--source_lang", type=str, default="Russian", help="Язык источника")
    parser.add_argument("--target_lang", type=str, default="English", help="Язык перевода")
    parser.add_argument("--api_url", type=str, default="http://localhost:1234/api/v1/chat", help="URL API LM Studio")
    parser.add_argument("--model", type=str, default="gigachat3.1-10b-a1.8b", help="Название модели")

    args = parser.parse_args()

    print("="*60)
    print("ОЦЕНКА КАЧЕСТВА ПЕРЕВОДА БЕЗ ЭТАЛОНА (Back-Translation)")
    print("="*60)

    # Инициализация
    client = LMStudioClient(base_url=args.api_url, model=args.model)

    # Загрузка данных
    print(f"\nЗагрузка датасета из {args.dataset}...")
    try:
        sources = load_dataset(args.dataset, args.samples)
        print(f"Загружено {len(sources)} текстов")
    except Exception as e:
        logger.error(f"Ошибка загрузки датасета: {e}")
        return

    if not sources:
        print("Нет текстов для обработки.")
        return

    # Этап 1: Прямой перевод (Source -> Target)
    print(f"\nЭтап 1: Выполнение прямого перевода ({args.source_lang} -> {args.target_lang})...")
    translations = []
    for i, text in enumerate(sources):
        print(f"  [{i+1}/{len(sources)}] Перевод: {text[:40]}...", end="")
        trans = client.translate(text, args.source_lang, args.target_lang)
        if trans:
            translations.append(trans)
            print(f" -> {trans[:40]}...")
        else:
            print(" [ERROR]")
            translations.append("") # Плейсхолдер, чтобы не сбить индексы, хотя лучше прервать

    # Фильтрация неудачных переводов
    valid_pairs = [(s, t) for s, t in zip(sources, translations) if t]
    if not valid_pairs:
        print("\nОшибка: не удалось получить ни одного перевода.")
        return

    sources_valid = [p[0] for p in valid_pairs]
    translations_valid = [p[1] for p in valid_pairs]
    print(f"\nУспешно переведено {len(valid_pairs)} текстов.")

    # Этап 2: Обратный перевод (Target -> Source)
    print(f"\nЭтап 2: Выполнение обратного перевода ({args.target_lang} -> {args.source_lang})...")
    back_translations = []
    for i, text in enumerate(translations_valid):
        print(f"  [{i+1}/{len(translations_valid)}] Обратный перевод: {text[:40]}...", end="")
        back_trans = client.back_translate(text, args.target_lang, args.source_lang)
        if back_trans:
            back_translations.append(back_trans)
            print(f" -> {back_trans[:40]}...")
        else:
            print(" [ERROR]")
            back_translations.append("")

    # Фильтрация неудачных обратных переводов
    final_data = [(s, t, b) for s, t, b in zip(sources_valid, translations_valid, back_translations) if b]
    if not final_data:
        print("\nОшибка: не удалось получить ни одного обратного перевода.")
        return

    sources_final = [d[0] for d in final_data]
    translations_final = [d[1] for d in final_data]
    back_translations_final = [d[2] for d in final_data]

    print(f"\nУспешно выполнено {len(final_data)} полных циклов перевода.")

    # Этап 3: Вычисление метрик (Source vs Back-Translation)
    print("\n" + "="*60)
    print("ВЫЧИСЛЕНИЕ МЕТРИК (Оригинал vs Обратный перевод)")
    print("="*60)

    metrics = calculate_metrics(sources_final, back_translations_final)

    # Вывод результатов
    print("\n" + "="*60)
    print("РЕЗУЛЬТАТЫ ОЦЕНКИ (Back-Translation Consistency)")
    print("="*60)
    print(f"{'Метрика':<25} {'Значение':>15}")
    print("-" * 45)
    print(f"BLEU Score:          {metrics['bleu']['score']:>15.4f}")
    print(f"ROUGE1 F1:           {metrics['rouge']['rouge1']:>15.4f}")
    print(f"ROUGE2 F1:           {metrics['rouge']['rouge2']:>15.4f}")
    print(f"ROUGEL F1:           {metrics['rouge']['rougeL']:>15.4f}")
    print(f"BERTScore Precision: {metrics['bertscore']['precision']:>15.4f}")
    print(f"BERTScore Recall:    {metrics['bertscore']['recall']:>15.4f}")
    print(f"BERTScore F1:        {metrics['bertscore']['f1']:>15.4f}")

    # Сохранение в JSON
    output_file = "results_no_reference.json"
    result_data = {
        "config": {
            "model": args.model,
            "source_lang": args.source_lang,
            "target_lang": args.target_lang,
            "method": "back_translation",
            "num_samples": len(final_data)
        },
        "metrics": metrics,
        "samples": [
            {
                "source": s,
                "translation": t,
                "back_translation": b
            }
            for s, t, b in final_data
        ]
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)

    print(f"\nРезультаты сохранены в {output_file}")


if __name__ == "__main__":
    main()