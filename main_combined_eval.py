#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Комбинированная оценка качества перевода:
1. С эталоном (Reference-based): COMET (нейросетевая метрика).
2. Без эталона (No-Reference): Semantic, Round-trip, NER, Perplexity.
3. Сравнение совокупных метрик и корреляция.
"""

import argparse
import json
import logging
import time
import requests
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import numpy as np

# --- Метрики ---
from evaluation_reference import ReferenceBasedEvaluator
from bert_score import score as bert_score

# --- Для NER и Perplexity ---
try:
    from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForCausalLM, pipeline, Pipeline
    import torch
    from collections import Counter
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# --- Настройка логирования ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("translation_eval.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==========================================
# КЛИЕНТ LM STUDIO
# ==========================================

class LMStudioClient:
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url
        self.model = model
        logger.info(f"Инициализация клиента. API endpoint: {base_url}")

    def translate(self, text: str, source_lang: str, target_lang: str, system_prompt_override: str = None) -> str:
        if system_prompt_override:
            system_prompt = system_prompt_override
        else:
            system_prompt = (
                f"You are a professional translator. Translate the following text from {source_lang} to {target_lang}. "
                "Return ONLY the translated text, without any explanations."
            )

        payload = {
            "model": self.model,
            "system_prompt": system_prompt,
            "input": text
        }

        try:
            response = requests.post(f"{self.base_url}", json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()

            # Поддержка разных форматов ответа
            if "output" in data and isinstance(data["output"], list) and len(data["output"]) > 0:
                content = data["output"][0].get("content", "")
                return content.strip()
            elif "response" in data:
                return data["response"].strip()
            else:
                logger.warning(f"Неизвестный формат ответа: {data}")
                return ""
        except Exception as e:
            logger.error(f"Ошибка при запросе к API: {e}")
            return ""

# ==========================================
# МОДУЛЬ ОЦЕНКИ С ЭТАЛОНОМ
# ==========================================

def evaluate_reference_based(sources: List[str], translations: List[str], references: List[str]) -> Dict[str, Any]:
    """
    Вычисляет метрики с эталоном используя COMET.
    """
    logger.info("Вычисление метрик с эталоном (COMET)...")

    # Инициализируем evaluator с COMET
    evaluator = ReferenceBasedEvaluator()

    # Вычисляем COMET
    results = evaluator.evaluate_all(sources, translations, references)

    comet_mean = results["comet"]["comet_mean"]

    # COMET уже является совокупной оценкой (она обучена предсказывать человеческую оценку)
    ref_aggregate = comet_mean

    return {
        "comet": comet_mean,
        "aggregate_score": ref_aggregate
    }

# ==========================================
# МОДУЛЬ ОЦЕНКИ БЕЗ ЭТАЛОНА (НОВЫЕ МЕТРИКИ)
# ==========================================

class NoReferenceEvaluator:
    def __init__(self, client: LMStudioClient, source_lang: str, target_lang: str):
        self.client = client
        self.source_lang = source_lang
        self.target_lang = target_lang

        # Загрузка моделей для NER (Perplexity теперь эвристический и не требует моделей)
        self.ner_model = None
        self.ner_tokenizer = None

        if TRANSFORMERS_AVAILABLE:
            self._load_ner_model()
        else:
            logger.warning("Transformers/torch не установлены. NER будет пропущен.")

    def _load_ner_model(self):
        """Загружает только NER модели, так как Perplexity теперь эвристический."""
        logger.info("Загрузка NER модели (это может занять время)...")
        try:
            # Используем русскую модель для NER, так как обратный перевод на русский
            # Модель "Gherman/bert-base-NER-Russian" - современная и точная модель
            ner_model_name = "Gherman/bert-base-NER-Russian"
            logger.info(f"Загрузка модели {ner_model_name}...")
            self.ner_tokenizer = AutoTokenizer.from_pretrained(ner_model_name)
            self.ner_model = pipeline("ner", model=ner_model_name, aggregation_strategy="simple")
            logger.info("NER модель загружена успешно.")
        except Exception as e:
            logger.error(f"Не удалось загрузить NER модель: {e}. Попробуем запасной вариант...")
            try:
                # Запасной вариант: sergeyzh/BERT-ner-ru
                ner_model_name = "sergeyzh/BERT-ner-ru"
                logger.info(f"Загрузка запасной модели {ner_model_name}...")
                self.ner_tokenizer = AutoTokenizer.from_pretrained(ner_model_name)
                self.ner_model = AutoModelForTokenClassification.from_pretrained(ner_model_name)
                logger.info("Запасная NER модель загружена успешно.")
            except Exception as e2:
                logger.error(f"Не удалось загрузить ни одну NER модель: {e2}. NER будет пропущен.")
                self.ner_model = None
                self.ner_tokenizer = None

    def extract_entities(self, text: str) -> set:
        """Извлекает именованные сущности из текста используя русскую NER модель."""
        if not self.ner_model or not text:
            return set()

        try:
            # Если используется pipeline (новая модель), вызываем напрямую
            if isinstance(self.ner_model, Pipeline):
                results = self.ner_model(text)
                entities = set()
                for result in results:
                    entity_text = result.get('entity_group', result.get('entity', ''))
                    if entity_text:
                        entities.add(entity_text.strip())
                return entities

            # Старый формат с tokenizer и model
            inputs = self.ner_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.ner_model(**inputs)

            predictions = torch.argmax(outputs.logits, dim=2)
            tokens = self.ner_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

            # Извлечение сущностей
            entities = set()
            current_entity = []

            # Маппинг ID в лейблы из модели
            id2label = self.ner_model.config.id2label

            for i, token_id in enumerate(inputs["input_ids"][0]):
                if token_id == self.ner_tokenizer.cls_token_id or token_id == self.ner_tokenizer.sep_token_id:
                    continue
                if i >= len(predictions[0]):
                    break

                pred_label = id2label.get(predictions[0][i].item(), 'O')
                if pred_label != 'O':
                    # Удаляем префиксы B-, I-
                    entity_type = pred_label.split('-')[-1]
                    token_str = tokens[i].replace('##', '').replace('▁', '')
                    current_entity.append(token_str)
                else:
                    if current_entity:
                        entities.add(" ".join(current_entity))
                        current_entity = []
            if current_entity:
                entities.add(" ".join(current_entity))
            return entities
        except Exception as e:
            logger.warning(f"NER extraction failed: {e}")
            return set()

    def calculate_perplexity(self, text: str) -> float:
        """
        Рассчитывает эвристическую перплексию (оценку гладкости текста) без использования тяжелых LM.
        Основано на:
        1. Повторении n-грамм (чем больше повторов, тем ниже качество / выше перплексия).
        2. Соотношении уникальных токенов к общей длине.
        3. Длине предложения (слишком короткие могут быть обрывками).

        Возвращает значение в диапазоне ~1.0 (отличный текст) до >50.0 (бессвязный набор слов).
        """
        if not text or len(text.strip()) < 2:
            return 100.0  # Высокая перплексия для пустых текстов

        tokens = text.lower().split()
        if len(tokens) == 0:
            return 100.0

        # 1. Оценка повторяемости (Unigram repetition penalty)
        unique_tokens = set(tokens)
        uniqueness_ratio = len(unique_tokens) / len(tokens)

        # 2. Bigram repetition (насколько часто повторяются пары слов)
        bigrams = [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens)-1)]
        if len(bigrams) > 0:
            unique_bigrams = set(bigrams)
            bigram_repetition = 1.0 - (len(unique_bigrams) / len(bigrams))
        else:
            bigram_repetition = 0.0

        # 3. Штраф за слишком короткие тексты (менее 3 слов считается ненадежным)
        length_penalty = 1.0
        if len(tokens) < 3:
            length_penalty = 2.0
        elif len(tokens) < 5:
            length_penalty = 1.5

        # Эвристический расчет "псевдо-перплексии"
        # Базовое значение + штрафы
        # Если текст уникальный (ratio ~1.0) и нет повторов биграмм, перплексия низкая (~1.0-5.0)
        # Если много повторов, растет экспоненциально

        base_ppl = 1.0
        # Штраф за неуникальность: если ratio=0.5, множитель = 2.0
        uniqueness_penalty = (1.0 / uniqueness_ratio) if uniqueness_ratio > 0 else 10.0

        # Штраф за повторы биграмм: если 50% повторов, добавляем значительный штраф
        bigram_penalty = 1.0 + (bigram_repetition * 10.0)

        calculated_ppl = base_ppl * uniqueness_penalty * bigram_penalty * length_penalty

        # Ограничиваем разумными пределами для интерпретации
        return min(max(calculated_ppl, 1.0), 500.0)

    def evaluate_no_reference(self, sources: List[str], translations: List[str]) -> Tuple[List[Dict], Dict]:
        logger.info("Выполнение оценки без эталона (Back-translation + Advanced Metrics)...")

        back_translations = []
        metrics_list = []

        # Этап 1: Обратный перевод
        logger.info("Этап 1: Обратный перевод...")
        for tr in tqdm(translations, desc="Back-translating"):
            # Переводим обратно на исходный язык
            back_tr = self.client.translate(
                tr,
                source_lang=self.target_lang,
                target_lang=self.source_lang
            )
            back_translations.append(back_tr)

        # Этап 2: Вычисление метрик для каждой пары
        logger.info("Этап 2: Вычисление продвинутых метрик...")

        semantic_scores = []
        roundtrip_scores = []
        ner_scores = []
        perplexity_scores = [] # Сырые значения

        for i in tqdm(range(len(sources)), desc="Calculating Metrics"):
            src = sources[i]
            back_tr = back_translations[i]
            original_tr = translations[i]

            # 1. Semantic Similarity (BERTScore Source vs Back-Translation)
            # Используем multilingual модель для сравнения RU и RU (или RU и EN если back_tr сломан)
            try:
                _, _, f1 = bert_score([back_tr], [src], lang="ru", verbose=False, rescale_with_baseline=True)
                sem_score = f1[0].item()
            except:
                sem_score = 0.0

            # 2. Round-trip Consistency (BLEU Source vs Back-Translation)
            # Если тексты пустые, BLEU не считается
            if src.strip() and back_tr.strip():
                try:
                    rt_bleu = corpus_bleu(
                        [[src.split()]],
                        [back_tr.split()],
                        smoothing_function=SmoothingFunction().method1
                    )
                except:
                    rt_bleu = 0.0
            else:
                rt_bleu = 0.0

            # 3. NER Consistency
            src_entities = self.extract_entities(src)
            back_entities = self.extract_entities(back_tr)

            if len(src_entities) == 0 and len(back_entities) == 0:
                ner_score = 1.0 # Нет сущностей - нет ошибок
            elif len(src_entities) == 0 or len(back_entities) == 0:
                ner_score = 0.0 # Одна сторона имеет сущности, другая нет
            else:
                # Jaccard similarity или F1 overlap
                intersection = src_entities.intersection(back_entities)
                union = src_entities.union(back_entities)
                ner_score = len(intersection) / len(union) if union else 0.0

            # 4. Cross-Entropy (Perplexity of Back-Translation)
            # Оцениваем, насколько "естественен" обратный перевод для родного языка
            ppl = self.calculate_perplexity(back_tr)
            # Нормализация perplexity в score (0-1).
            # Типичная перплексия хорошего текста ~10-50. Плохого >100.
            # Формула: score = 1 / (1 + ln(ppl)) если ppl > 1 else 1
            if ppl == float('inf') or ppl <= 0:
                ppl_score = 0.0
            else:
                import math
                ppl_score = 1.0 / (1.0 + math.log(ppl)) if ppl > 1 else 1.0

            semantic_scores.append(sem_score)
            roundtrip_scores.append(rt_bleu)
            ner_scores.append(ner_score)
            perplexity_scores.append(ppl_score)

            metrics_list.append({
                "semantic": sem_score,
                "roundtrip_bleu": rt_bleu,
                "ner_consistency": ner_score,
                "perplexity_score": ppl_score,
                "back_translation": back_tr
            })

        # Агрегация
        avg_semantic = np.mean(semantic_scores)
        avg_roundtrip = np.mean(roundtrip_scores)
        avg_ner = np.mean(ner_scores)
        avg_ppl = np.mean(perplexity_scores)

        # Совокупная оценка (No-Ref Aggregate)
        # Увеличиваем вес семантики и perplexity, снижаем вес round-trip BLEU
        # Round-trip BLEU слишком чувствителен к формулировкам
        weights = {"semantic": 0.50, "roundtrip": 0.10, "ner": 0.20, "perplexity": 0.20}
        no_ref_aggregate = (
                avg_semantic * weights["semantic"] +
                avg_roundtrip * weights["roundtrip"] +
                avg_ner * weights["ner"] +
                avg_ppl * weights["perplexity"]
        )

        summary = {
            "semantic_similarity": avg_semantic,
            "roundtrip_consistency": avg_roundtrip,
            "ner_consistency": avg_ner,
            "perplexity_score": avg_ppl,
            "aggregate_score": no_ref_aggregate
        }

        return metrics_list, summary

# ==========================================
# ОСНОВНАЯ ЛОГИКА
# ==========================================

def load_dataset(path: str, limit: int = None) -> Tuple[List[str], List[str]]:
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

def main():
    parser = argparse.ArgumentParser(description="Combined Translation Evaluation")
    parser.add_argument("--samples", type=int, default=5, help="Количество образцов")
    parser.add_argument("--dataset", type=str, default="datasets/rus-eng-part1-text.txt", help="Путь к датасету")
    parser.add_argument("--api_url", type=str, default="http://localhost:1234/api/v1/chat", help="URL API LM Studio")
    parser.add_argument("--model", type=str, default="gigachat3.1-10b-a1.8b", help="Название модели")
    args = parser.parse_args()

    print("="*60)
    print("КОМБИНИРОВАННАЯ ОЦЕНКА КАЧЕСТВА ПЕРЕВОДА")
    print("="*60)

    # 1. Загрузка данных
    sources, references = load_dataset(args.dataset, args.samples)
    if not sources:
        logger.error("Датасет пуст или не найден.")
        return

    # 2. Инициализация клиента
    client = LMStudioClient(args.api_url, args.model)

    # 3. Прямой перевод (нужен для обоих типов оценки)
    print("\nВыполнение прямого перевода (Source -> Target)...")
    translations = []
    for i, src in enumerate(tqdm(sources, desc="Translating")):
        tr = client.translate(src, source_lang="Russian", target_lang="English")
        translations.append(tr)
        print(f"  [{i+1}] {src[:30]}... -> {tr[:30]}...")

    # 4. Оценка с эталоном
    print("\n" + "="*40)
    print("ЧАСТЬ 1: ОЦЕНКА С ЭТАЛОНОМ")
    print("="*40)
    ref_metrics = evaluate_reference_based(sources, translations, references)

    print(f"COMET (основная метрика): {ref_metrics['comet']:.4f}")
    print(f"--> Совокупная оценка (Ref): {ref_metrics['aggregate_score']:.4f}")

    # 5. Оценка без эталона (включая Back-translation и новые метрики)
    print("\n" + "="*40)
    print("ЧАСТЬ 2: ОЦЕНКА БЕЗ ЭТАЛОНА (Advanced)")
    print("="*40)
    evaluator = NoReferenceEvaluator(client, "Russian", "English")
    no_ref_details, no_ref_metrics = evaluator.evaluate_no_reference(sources, translations)

    print(f"Semantic Similarity: {no_ref_metrics['semantic_similarity']:.4f}")
    print(f"Round-trip Consistency: {no_ref_metrics['roundtrip_consistency']:.4f}")
    print(f"NER Consistency: {no_ref_metrics['ner_consistency']:.4f}")
    print(f"Perplexity Score: {no_ref_metrics['perplexity_score']:.4f}")
    print(f"--> Совокупная оценка (No-Ref): {no_ref_metrics['aggregate_score']:.4f}")

    # 6. Корреляция и выводы
    print("\n" + "="*40)
    print("ИТОГОВОЕ СРАВНЕНИЕ")
    print("="*40)
    print(f"Совокупная оценка (Ref Based)   : {ref_metrics['aggregate_score']:.4f}")
    print(f"Совокупная оценка (No-Ref Based): {no_ref_metrics['aggregate_score']:.4f}")

    # Попытка вычислить корреляцию по выборке (если бы у нас было много точек)
    # Здесь у нас одна точка (среднее по выборке). Для реальной корреляции нужно много запусков с разными моделями.
    # Но мы можем показать разрыв (Gap).
    gap = abs(ref_metrics['aggregate_score'] - no_ref_metrics['aggregate_score'])
    print(f"Разрыв между оценками (Gap)     : {gap:.4f}")

    if gap < 0.1:
        print("Вердикт: Метрики хорошо согласуются!")
    else:
        print("Вердикт: Метрики расходятся. Требуется калибровка весов.")

    # Сохранение результатов
    results = {
        "config": vars(args),
        "reference_evaluation": ref_metrics,
        "no_reference_evaluation": no_ref_metrics,
        "samples": [
            {
                "source": sources[i],
                "translation": translations[i],
                "reference": references[i],
                "back_translation": no_ref_details[i]["back_translation"],
                "scores": no_ref_details[i]
            }
            for i in range(len(sources))
        ]
    }

    output_file = "results_combined.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nРезультаты сохранены в {output_file}")

if __name__ == "__main__":
    main()