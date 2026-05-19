#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Эксперимент по проверке корреляции безэталонных метрик с эталонной метрикой COMET.

Постановка эксперимента (согласно требованиям научного руководителя):
1. На исходных текстах генерируются гипотезы разными провайдерами (или одной моделью
   с разными параметрами температуры) - это создаёт разброс качества перевода
2. Для каждой гипотезы считается COMET по триплету (исходник, гипотеза, эталон) -
   это эталонная оценка качества
3. Для каждой гипотезы считается агрегированная безэталонная оценка (semantic +
   roundtrip + NER + perplexity) - это наша методика без эталона
4. Проверяется корреляция Спирмена/Пирсона между двумя рядами оценок

Ключевое отличие от предыдущей версии:
- Мы оцениваем не качество эталона, а способность безэталонных метрик ловить разброс
  качества между разными гипотезами (переводами от разных провайдеров/температур)
- Объектом оценки на проде является гипотеза (перевод), а не эталон
- Если корреляция высокая, значит наши безэталонные метрики могут заменить COMET
  в продакшене, где нет эталонов

Результат сохраняется в Excel и CSV файлы для последующего анализа и построения графиков.
Выходной файл содержит:
- All_Results: полные данные по каждому переводу
- Summary: сводная статистика и значения корреляций
- By_Temperature: статистика по каждой температуре
- Correlation_Matrix: матрица корреляций всех метрик
"""

import argparse
import json
import logging
import random
import time
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

# --- Метрики ---
from evaluation_reference import ReferenceBasedEvaluator
from bert_score import score as bert_score
import sacrebleu
import math

# --- Для NER ---
try:
    from transformers import AutoTokenizer, pipeline, Pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# --- Настройка логирования ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("experiment_correlation.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==========================================
# КЛИЕНТ LM STUDIO
# ==========================================

class LMStudioClient:
    def __init__(self, base_url: str, model: str, temperature: float = 0.7, top_p: float = 0.9):
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        logger.info(f"Инициализация клиента. API endpoint: {base_url}, модель: {model}")

    def translate(self, text: str, source_lang: str, target_lang: str,
                  system_prompt_override: str = None, temperature: Optional[float] = None) -> str:
        if system_prompt_override:
            system_prompt = system_prompt_override
        else:
            system_prompt = (
                f"You are a professional translator. Translate the following text from {source_lang} to {target_lang}. "
                "Return ONLY the translated text, without any explanations."
            )

        # Используем переданную температуру или температуру по умолчанию
        temp = temperature if temperature is not None else self.temperature

        payload = {
            "model": self.model,
            "system_prompt": system_prompt,
            "input": text,
            "temperature": temp,
            "top_p": self.top_p
        }

        try:
            import requests
            response = requests.post(f"{self.base_url}", json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()

            # Поддержка разных форматов ответа
            if "output" in data and isinstance(data["output"], list) and len(data["output"]) > 0:
                content = data["output"][0].get("content", "")
                return content.strip()
            elif "response" in data:
                return data["response"].strip()
            elif "choices" in data and len(data["choices"]) > 0:
                return data["choices"][0].get("message", {}).get("content", "").strip()
            else:
                logger.warning(f"Неизвестный формат ответа: {data}")
                return ""
        except Exception as e:
            logger.error(f"Ошибка при запросе к API: {e}")
            return ""

# ==========================================
# МОДУЛЬ ОЦЕНКИ С ЭТАЛОНОМ (COMET)
# ==========================================

def calculate_comet_score(sources: List[str], translations: List[str],
                          references: List[str], evaluator: ReferenceBasedEvaluator) -> List[float]:
    """
    Вычисляет COMET score для каждой пары переводов.
    Возвращает список оценок (по одной на каждый перевод).
    """
    logger.info("Вычисление COMET оценок...")

    results = evaluator.calculate_comet(sources, translations, references)

    # Возвращаем индивидуальные оценки для каждого перевода
    return results["comet_scores"]

# ==========================================
# МОДУЛЬ ОЦЕНКИ БЕЗ ЭТАЛОНА
# ==========================================

class NoReferenceEvaluator:
    def __init__(self, client: LMStudioClient, source_lang: str, target_lang: str,
                 weights: Dict[str, float] = None):
        self.client = client
        self.source_lang = source_lang
        self.target_lang = target_lang

        # Веса метрик (можно настроить через валидацию)
        self.weights = weights or {
            "semantic": 0.30,
            "roundtrip": 0.40,
            "ner": 0.15,
            "perplexity": 0.15
        }

        # Загрузка моделей для NER
        self.ner_model = None
        self.ner_tokenizer = None

        if TRANSFORMERS_AVAILABLE:
            self._load_ner_model()
        else:
            logger.warning("Transformers/torch не установлены. NER будет пропущен.")

    def _load_ner_model(self):
        """Загружает NER модели."""
        logger.info("Загрузка NER модели (это может занять время)...")
        try:
            ner_model_name = "Gherman/bert-base-NER-Russian"
            logger.info(f"Загрузка модели {ner_model_name}...")
            self.ner_tokenizer = AutoTokenizer.from_pretrained(ner_model_name)
            self.ner_model = pipeline("ner", model=ner_model_name, aggregation_strategy="simple")
            logger.info("NER модель загружена успешно.")
        except Exception as e:
            logger.error(f"Не удалось загрузить NER модель: {e}. Попробуем запасной вариант...")
            try:
                ner_model_name = "sergeyzh/BERT-ner-ru"
                logger.info(f"Загрузка запасной модели {ner_model_name}...")
                self.ner_tokenizer = AutoTokenizer.from_pretrained(ner_model_name)
                self.ner_model = pipeline("ner", model=ner_model_name, aggregation_strategy="simple")
                logger.info("Запасная NER модель загружена успешно.")
            except Exception as e2:
                logger.error(f"Не удалось загрузить ни одну NER модель: {e2}. NER будет пропущен.")
                self.ner_model = None
                self.ner_tokenizer = None

    def extract_entities(self, text: str) -> set:
        """Извлекает именованные сущности из текста."""
        if not self.ner_model or not text:
            return set()

        try:
            if isinstance(self.ner_model, Pipeline):
                results = self.ner_model(text)
                entities = set()
                for result in results:
                    entity_text = result.get('entity_group', result.get('entity', ''))
                    if entity_text:
                        entities.add(entity_text.strip())
                return entities
        except Exception as e:
            logger.warning(f"NER extraction failed: {e}")
            return set()

    def calculate_perplexity(self, text: str) -> float:
        """
        Рассчитывает эвристическую перплексию (оценку гладкости текста).
        """
        if not text or len(text.strip()) < 2:
            return 100.0

        tokens = text.lower().split()
        if len(tokens) == 0:
            return 100.0

        unique_tokens = set(tokens)
        uniqueness_ratio = len(unique_tokens) / len(tokens)

        bigrams = [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens)-1)]
        if len(bigrams) > 0:
            unique_bigrams = set(bigrams)
            bigram_repetition = 1.0 - (len(unique_bigrams) / len(bigrams))
        else:
            bigram_repetition = 0.0

        length_penalty = 1.0
        if len(tokens) < 3:
            length_penalty = 2.0
        elif len(tokens) < 5:
            length_penalty = 1.5

        base_ppl = 1.0
        uniqueness_penalty = (1.0 / uniqueness_ratio) if uniqueness_ratio > 0 else 10.0
        bigram_penalty = 1.0 + (bigram_repetition * 10.0)

        calculated_ppl = base_ppl * uniqueness_penalty * bigram_penalty * length_penalty
        return min(max(calculated_ppl, 1.0), 500.0)

    def evaluate_single_translation(self, source: str, translation: str,
                                    back_translation: str = None) -> Dict[str, float]:
        """
        Вычисляет все безэталонные метрики для одного перевода.

        Args:
            source: Исходный текст
            translation: Перевод (гипотеза)
            back_translation: Обратный перевод (если уже вычислен)

        Returns:
            Словарь с индивидуальными метриками и aggregate_score
        """
        # 1. Обратный перевод (если не предоставлен)
        if back_translation is None:
            back_translation = self.client.translate(
                translation,
                source_lang=self.target_lang,
                target_lang=self.source_lang
            )

        # 2. Semantic Similarity (BERTScore Source vs Back-Translation)
        try:
            _, _, f1 = bert_score([back_translation], [source], lang="ru",
                                  verbose=False, rescale_with_baseline=True)
            sem_score = f1[0].item()
        except:
            sem_score = 0.0

        # 3. Round-trip Consistency (chrF Source vs Back-Translation)
        if source.strip() and back_translation.strip():
            try:
                chrf_score = sacrebleu.corpus_chrf(
                    [back_translation],
                    [[source]],
                    beta=2,
                    word_order=2
                )
                rt_score = chrf_score.score / 100.0
            except Exception as e:
                rt_score = 0.0
        else:
            rt_score = 0.0

        # 4. NER Consistency
        src_entities = self.extract_entities(source)
        back_entities = self.extract_entities(back_translation)

        if len(src_entities) == 0 and len(back_entities) == 0:
            ner_score = 1.0
        elif len(src_entities) == 0 or len(back_entities) == 0:
            ner_score = 0.0
        else:
            intersection = src_entities.intersection(back_entities)
            union = src_entities.union(back_entities)
            ner_score = len(intersection) / len(union) if union else 0.0

        # 5. Perplexity Score
        ppl = self.calculate_perplexity(back_translation)
        if ppl == float('inf') or ppl <= 0:
            ppl_score = 0.0
        else:
            ppl_score = 1.0 / (1.0 + math.log(ppl)) if ppl > 1 else 1.0

        # Агрегированная оценка
        aggregate_score = (
                sem_score * self.weights["semantic"] +
                rt_score * self.weights["roundtrip"] +
                ner_score * self.weights["ner"] +
                ppl_score * self.weights["perplexity"]
        )

        return {
            "semantic": sem_score,
            "roundtrip_chrf": rt_score,
            "ner_consistency": ner_score,
            "perplexity_score": ppl_score,
            "aggregate_score": aggregate_score,
            "back_translation": back_translation
        }

# ==========================================
# ЗАГРУЗКА ДАННЫХ
# ==========================================

def load_dataset(path: str, limit: int = None) -> Tuple[List[str], List[str]]:
    """
    Загружает датасет формата: исходный текст ||| эталонный перевод
    """
    sources = []
    references = []
    logger.info(f"Загрузка датасета из {path}...")

    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    count = 0
    for line in lines:
        if '|||' in line:
            parts = line.strip().split('|||')
            if len(parts) >= 2:
                sources.append(parts[0].strip())
                references.append(parts[1].strip())
                count += 1
                if limit and count >= limit:
                    break

    logger.info(f"Загружено {count} пар.")
    return sources, references

# ==========================================
# ОСНОВНОЙ ЭКСПЕРИМЕНТ
# ==========================================

def run_experiment(args):
    """
    Основной эксперимент по проверке корреляции метрик.

    План:
    1. Для каждого текста в выборке генерируем несколько переводов (разные температуры/провайдеры)
    2. Для каждого перевода считаем COMET (с эталоном)
    3. Для каждого перевода считаем безэталонные метрики
    4. Сохраняем все оценки в DataFrame
    5. Вычисляем корреляцию между COMET и aggregate_score
    6. Экспортируем в Excel
    """

    print("="*80)
    print("ЭКСПЕРИМЕНТ: КОРРЕЛЯЦИЯ БЕЗЭТАЛОННЫХ МЕТОДИК С COMET")
    print("="*80)

    # 1. Загрузка данных
    sources, references = load_dataset(args.dataset, args.samples)
    if not sources:
        logger.error("Датасет пуст или не найден.")
        return

    # 2. Инициализация клиента (провайдер переводов)
    # Можно менять model и temperature для разных провайдеров
    client = LMStudioClient(
        base_url=args.api_url,
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p
    )

    # 3. Инициализация оценщиков
    ref_evaluator = ReferenceBasedEvaluator()
    no_ref_evaluator = NoReferenceEvaluator(
        client,
        source_lang="Russian",
        target_lang="English",
        weights=args.weights
    )

    # 4. Генерация переводов с разными параметрами для создания разброса качества
    # Используем температуры из аргументов (по умолчанию [1.0, 0.8, 0.6, 0.4])
    temperatures = args.temperatures

    print(f"\nГенерация переводов с температурами: {temperatures}")
    print(f"Количество текстов: {len(sources)}")
    print(f"Общее количество переводов: {len(sources) * len(temperatures)}")

    # Структура для хранения результатов
    all_results = []

    # 5. Генерация переводов и оценка
    for temp_idx, temperature in enumerate(temperatures):
        print(f"\n{'='*60}")
        print(f"Генерация переводов с температурой {temperature} ({temp_idx+1}/{len(temperatures)})")
        print(f"{'='*60}")

        # Создаем временный клиент с этой температурой
        temp_client = LMStudioClient(
            base_url=args.api_url,
            model=args.model,
            temperature=temperature,
            top_p=args.top_p
        )

        # Обновляем клиент в no_ref_evaluator
        no_ref_evaluator.client = temp_client

        translations = []

        # Генерация переводов для всех текстов
        for i, src in enumerate(tqdm(sources, desc=f"Translating (T={temperature})")):
            tr = temp_client.translate(
                src,
                source_lang="Russian",
                target_lang="English",
                temperature=temperature
            )
            translations.append(tr)

            if i < 3:  # Показать первые 3 примера
                print(f"  [{i+1}] {src[:40]}... -> {tr[:40]}...")

        # 6. Оценка с эталоном (COMET) для всех переводов этой температуры
        print(f"\nВычисление COMET оценок для температуры {temperature}...")
        comet_scores = calculate_comet_score(sources, translations, references, ref_evaluator)

        # 7. Оценка без эталона для каждого перевода
        print(f"Вычисление безэталонных метрик для температуры {temperature}...")
        for i in tqdm(range(len(sources)), desc=f"No-Ref Metrics (T={temperature})"):
            src = sources[i]
            tr = translations[i]
            ref = references[i]
            comet = comet_scores[i]

            # Вычисляем безэталонные метрики
            no_ref_metrics = no_ref_evaluator.evaluate_single_translation(src, tr)

            # Сохраняем результат
            result = {
                "sample_id": i,
                "temperature": temperature,
                "provider_config": f"{args.model}_T{temperature}",
                "source_text": src,
                "translation": tr,
                "reference": ref,
                "back_translation": no_ref_metrics["back_translation"],
                "comet_score": comet,
                "semantic_score": no_ref_metrics["semantic"],
                "roundtrip_chrf": no_ref_metrics["roundtrip_chrf"],
                "ner_consistency": no_ref_metrics["ner_consistency"],
                "perplexity_score": no_ref_metrics["perplexity_score"],
                "aggregate_no_ref": no_ref_metrics["aggregate_score"]
            }

            all_results.append(result)

    # 8. Создание DataFrame
    df = pd.DataFrame(all_results)

    # 9. Вычисление корреляций
    print("\n" + "="*80)
    print("РЕЗУЛЬТАТЫ КОРРЕЛЯЦИИ")
    print("="*80)

    # Корреляция между COMET и aggregate_no_ref
    comet_vals = df["comet_score"].values
    agg_vals = df["aggregate_no_ref"].values

    # Пирсон (линейная корреляция)
    pearson_corr, pearson_pvalue = pearsonr(comet_vals, agg_vals)

    # Спирмен (ранговая корреляция, более устойчива к выбросам)
    spearman_corr, spearman_pvalue = spearmanr(comet_vals, agg_vals)

    print(f"\nКорреляция между COMET и Aggregate No-Ref:")
    print(f"  Пирсон: r = {pearson_corr:.4f}, p-value = {pearson_pvalue:.6f}")
    print(f"  Спирмен: ρ = {spearman_corr:.4f}, p-value = {spearman_pvalue:.6f}")

    # Корреляция для отдельных метрик
    print(f"\nКорреляция отдельных метрик с COMET:")
    for metric in ["semantic_score", "roundtrip_chrf", "ner_consistency", "perplexity_score"]:
        p_corr, p_pval = pearsonr(df[metric].values, comet_vals)
        s_corr, s_pval = spearmanr(df[metric].values, comet_vals)
        print(f"  {metric}:")
        print(f"    Пирсон: r = {p_corr:.4f}, p-value = {p_pval:.6f}")
        print(f"    Спирмен: ρ = {s_corr:.4f}, p-value = {s_pval:.6f}")

    # 10. Статистика по температурам
    print(f"\nСтатистика по температурам:")
    temp_stats = df.groupby("temperature").agg({
        "comet_score": ["mean", "std", "min", "max"],
        "aggregate_no_ref": ["mean", "std", "min", "max"]
    }).round(4)
    print(temp_stats)

    # 11. Сохранение в Excel и CSV
    output_excel = args.output or "experiment_correlation_results.xlsx"
    output_csv = output_excel.replace('.xlsx', '.csv')

    print(f"\nСохранение результатов в {output_excel} и {output_csv}...")

    # Создаем Excel с несколькими листами
    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        # Основные данные
        df.to_excel(writer, sheet_name='All_Results', index=False)

        # Сводная статистика
        summary_data = {
            "Metric": ["COMET Mean", "COMET Std", "Aggregate Mean", "Aggregate Std",
                       "Pearson Correlation", "Spearman Correlation"],
            "Value": [
                float(df["comet_score"].mean()),
                float(df["comet_score"].std()),
                float(df["aggregate_no_ref"].mean()),
                float(df["aggregate_no_ref"].std()),
                float(pearson_corr),
                float(spearman_corr)
            ],
            "P-Value": [
                None, None, None, None,
                float(pearson_pvalue),
                float(spearman_pvalue)
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

        # Статистика по температурам
        temp_summary = df.groupby("temperature").agg({
            "comet_score": ["mean", "std", "min", "max", "count"],
            "aggregate_no_ref": ["mean", "std", "min", "max"],
            "semantic_score": "mean",
            "roundtrip_chrf": "mean",
            "ner_consistency": "mean",
            "perplexity_score": "mean"
        }).round(4)
        temp_summary.to_excel(writer, sheet_name='By_Temperature')

        # Корреляционная матрица
        corr_cols = ["comet_score", "aggregate_no_ref", "semantic_score",
                     "roundtrip_chrf", "ner_consistency", "perplexity_score"]
        corr_matrix = df[corr_cols].corr()
        corr_matrix.to_excel(writer, sheet_name='Correlation_Matrix')

    # Дополнительно сохраняем CSV для удобства работы
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')

    print(f"Результаты сохранены в {output_excel}")

    # 12. Сохранение JSON для дополнительного анализа
    output_json = "experiment_correlation_results.json"
    experiment_summary = {
        "config": vars(args),
        "correlations": {
            "pearson": {
                "coefficient": float(pearson_corr),
                "p_value": float(pearson_pvalue)
            },
            "spearman": {
                "coefficient": float(spearman_corr),
                "p_value": float(spearman_pvalue)
            }
        },
        "statistics": {
            "total_samples": len(sources),
            "total_translations": len(all_results),
            "temperatures_used": temperatures,
            "comet_mean": float(df["comet_score"].mean()),
            "comet_std": float(df["comet_score"].std()),
            "aggregate_mean": float(df["aggregate_no_ref"].mean()),
            "aggregate_std": float(df["aggregate_no_ref"].std())
        }
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(experiment_summary, f, ensure_ascii=False, indent=2)

    print(f"Сводка эксперимента сохранена в {output_json}")

    print("\n" + "="*80)
    print("ЭКСПЕРИМЕНТ ЗАВЕРШЕН")
    print("="*80)

    return df, pearson_corr, spearman_corr

# ==========================================
# НАСТРОЙКА И ЗАПУСК
# ==========================================

def parse_args():
    parser = argparse.ArgumentParser(description="Experiment: Correlation of No-Ref metrics with COMET")

    # Параметры данных
    parser.add_argument("--samples", type=int, default=50,
                        help="Количество образцов из датасета (рекомендуется 50-100)")
    parser.add_argument("--dataset", type=str, default="datasets/rus-eng-part1-text.txt",
                        help="Путь к датасету")

    # Параметры API
    parser.add_argument("--api_url", type=str, default="http://localhost:1234/api/v1/chat",
                        help="URL API LM Studio")
    parser.add_argument("--model", type=str, default="gigachat3.1-10b-a1.8b",
                        help="Название модели для перевода")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Базовая температура (переопределяется --temperatures)")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p sampling parameter")

    # Параметры эксперимента
    parser.add_argument("--temperatures", type=float, nargs='+',
                        default=[1.0, 0.8, 0.6, 0.4],
                        help="Список температур для генерации разброса качества (по умолчанию: 1.0 0.8 0.6 0.4)")

    # Веса метрик
    parser.add_argument("--weights", type=json.loads,
                        default='{"semantic": 0.30, "roundtrip": 0.40, "ner": 0.15, "perplexity": 0.15}',
                        help="Веса метрик в формате JSON")

    # Вывод
    parser.add_argument("--output", type=str, default=None,
                        help="Имя выходного Excel файла")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_experiment(args)