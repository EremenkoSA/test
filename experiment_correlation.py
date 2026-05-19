import os
import json
import time
import random
import torch
import pandas as pd
from tqdm import tqdm
from translate import Translator # Ваш класс переводчика
from providers.config_loader import load_provider_config # Ваша функция загрузки конфига
from quality_assessor import QualityAssessor # Ваш класс оценки (будет использован частично или переопределен ниже)

# --- ИМПОРТЫ ДЛЯ НОВЫХ МЕТРИК ---
from bert_score import score as bert_score_func
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification, pipeline
import sacrebleu
import numpy as np

# КОНФИГУРАЦИЯ ЭКСПЕРИМЕНТА
TEMPERATURES = [1.0, 0.8, 0.6, 0.4]
SAMPLE_SIZE = 50  # Уменьшено для скорости теста, можно поставить 500
DATASET_PATH = "data/dataset.txt" # Путь к вашему датасету
OUTPUT_EXCEL = "results/correlation_experiment.xlsx"
OUTPUT_CSV = "results/correlation_experiment.csv"

# Проверка наличия CUDA
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# --- ЗАГРУЗКА МОДЕЛЕЙ ДЛЯ МЕТРИК (ОДИН РАЗ ПРИ СТАРТЕ) ---

print("Loading BERTScore model...")
# Используем мультиязычную модель для сравнения RU и EN
BERT_MODEL_TYPE = "bert-base-multilingual-cased"

print("Loading NER models...")
# Модель для русского
try:
    ner_ru_pipeline = pipeline("ner", model="blanchefort/rubert-base-cased-ner-mixed", tokenizer="blanchefort/rubert-base-cased-ner-mixed", device=0 if DEVICE == "cuda" else -1, aggregation_strategy="simple")
except Exception as e:
    print(f"Warning: Could not load RU NER model: {e}. Using dummy.")
    ner_ru_pipeline = None

# Модель для английского
try:
    ner_en_pipeline = pipeline("ner", model="dslim/bert-base-NER", tokenizer="dslim/bert-base-NER", device=0 if DEVICE == "cuda" else -1, aggregation_strategy="simple")
except Exception as e:
    print(f"Warning: Could not load EN NER model: {e}. Using dummy.")
    ner_en_pipeline = None

print("Loading GPT-2 for Perplexity...")
# GPT-2 для оценки перплексии английского текста
ppl_tokenizer = AutoTokenizer.from_pretrained("gpt2")
ppl_model = AutoModelForCausalLM.from_pretrained("gpt2").to(DEVICE)
ppl_tokenizer.pad_token = ppl_tokenizer.eos_token

def calculate_perplexity(text):
    """Расчет перплексии текста с помощью GPT-2"""
    try:
        encodings = ppl_tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
        with torch.no_grad():
            outputs = ppl_model(**encodings, labels=encodings.input_ids)
            loss = outputs.loss
        return float(torch.exp(loss))
    except Exception as e:
        print(f"Error calculating perplexity: {e}")
        return 9999.0 # Штраф за ошибку

def extract_entities_ru(text):
    if ner_ru_pipeline is None:
        return set()
    try:
        entities = ner_ru_pipeline(text)
        return set([e['word'].lower().strip() for e in entities if e['score'] > 0.5])
    except:
        return set()

def extract_entities_en(text):
    if ner_en_pipeline is None:
        return set()
    try:
        entities = ner_en_pipeline(text)
        return set([e['word'].lower().strip() for e in entities if e['score'] > 0.5])
    except:
        return set()

def calculate_jaccard(set1, set2):
    if not set1 and not set2:
        return 1.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0

def calculate_bert_score_multiling(src, hyp):
    """Считает BERTScore между исходником (RU) и гипотезой (EN)"""
    try:
        # rescale_with_baseline=True возвращает значения ~0-1, удобные для интерпретации
        P, R, F1 = bert_score_func([hyp], [src], lang="en", model_type=BERT_MODEL_TYPE, verbose=False, rescale_with_baseline=True)
        return float(F1[0])
    except Exception as e:
        print(f"BERTScore error: {e}")
        return 0.0

def calculate_chrf(source, back_translation):
    """Считает chrF++ между источником и обратным переводом"""
    if not back_translation:
        return 0.0
    try:
        # sacrebleu expects list of strings
        score = sacrebleu.corpus_chrf([back_translation], [[source]], lowercase=False, beta=3)
        return score.score / 100.0 # Нормализация к 0-1
    except Exception as e:
        print(f"chrF error: {e}")
        return 0.0

# --- ОСНОВНАЯ ЛОГИКА ЭКСПЕРИМЕНТА ---

def load_dataset(path, n_samples):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Фильтрация пустых строк и парсинг
    valid_lines = []
    for line in lines:
        if '|||' in line:
            valid_lines.append(line.strip())

    if len(valid_lines) < n_samples:
        print(f"Warning: Dataset has only {len(valid_lines)} lines, using all.")
        n_samples = len(valid_lines)

    selected_lines = random.sample(valid_lines, n_samples)

    for line in selected_lines:
        parts = line.split('|||')
        if len(parts) >= 2:
            en_text = parts[0].strip()
            ru_ref = parts[1].strip()
            data.append({"source_en": en_text, "reference_ru": ru_ref})
        elif len(parts) == 1:
            # Если нет эталона, пропускаем или используем как есть (зависит от формата)
            # В описании сказано: английский ||| русский эталон
            pass
    return data

def run_experiment():
    # 1. Загрузка данных
    print("Loading dataset...")
    dataset = load_dataset(DATASET_PATH, SAMPLE_SIZE)

    results = []

    # Конфигурация провайдера (берем одну, как в задании)
    # Предполагаем, что у вас есть способ загрузить конфиг по имени или пути
    # Здесь жестко зададим имя конфига, которое вы меняете вручную перед запуском
    provider_name = "gigachat3.1-10b-a1.8b" # Имя базового конфига

    for temp in TEMPERATURES:
        print(f"\n--- Running for Temperature: {temp} ---")

        # Формируем имя конфига или передаем параметр температуры в провайдер
        # В вашем коде, вероятно, нужно создать экземпляр провайдера с конкретной температурой
        # Пример: config = load_provider_config(provider_name)
        # config['temperature'] = temp

        # Имитация загрузки конфига с температурой (адаптируйте под свою структуру)
        try:
            # Попытка загрузить конкретный конфиг, если они названы по шаблону
            config_path = f"providers/{provider_name}_T{temp}.json"
            if os.path.exists(config_path):
                provider_config = load_provider_config(config_path.replace(".json", ""))
            else:
                # Если отдельного конфига нет, загружаем базовый и модифицируем температуру
                # Это зависит от реализации вашего ProviderWorker
                provider_config = load_provider_config(provider_name)
                if hasattr(provider_config, 'temperature'):
                    provider_config.temperature = temp
                elif isinstance(provider_config, dict):
                    provider_config['temperature'] = temp
        except Exception as e:
            print(f"Error loading config for temp {temp}: {e}")
            continue

        # Инициализация переводчика для этой температуры
        # Предполагается, что класс Translator принимает конфиг
        translator = Translator(provider_config=provider_config)

        for idx, item in enumerate(tqdm(dataset, desc=f"Temp {temp}")):
            source_text = item["source_en"] # Английский оригинал
            reference = item["reference_ru"] # Русский эталон

            # 1. Получение перевода (Гипотеза)
            try:
                # Вызов метода перевода. Адаптируйте под ваш API
                # Если Translator возвращает объект, возьмите .text
                translation_result = translator.translate(source_text, target_lang="ru")
                hypothesis = translation_result if isinstance(translation_result, str) else translation_result.text
            except Exception as e:
                print(f"Translation failed: {e}")
                hypothesis = ""

            if not hypothesis:
                continue

            # 2. Обратный перевод (для Round-Trip)
            try:
                # Создаем временный переводчик EN->RU->EN? Нет, у нас RU->EN.
                # Значит нужен переводчик RU->EN для обратного пути?
                # Исходник: EN. Гипотеза: RU. Обратный перевод: RU -> EN.
                # Нам нужен провайдер, который переводит с RU на EN.
                # Если у вас один провайдер двунаправленный, используем его.
                # Если нет, нужно создать второй инстанс или использовать тот же с swapped langs.

                # Предположим, что translator может перевести обратно
                back_translation_result = translator.translate(hypothesis, target_lang="en")
                back_translation = back_translation_result if isinstance(back_translation_result, str) else back_translation_result.text
            except Exception as e:
                back_translation = ""

            # 3. Расчет метрик

            # A. COMET (Эталонная оценка) - требует (Source, Hypothesis, Reference)
            # Source: EN, Hypothesis: RU, Reference: RU
            # Используем вашу существующую логику COMET, если она есть в QualityAssessor
            # Или вызываем напрямую, если есть глобальная модель
            comet_score = 0.0
            try:
                # Пример вызова, если у вас есть глобальный comet_model
                # from comet import download_model, load_from_checkpoint
                # comet_model = load_from_checkpoint("Unbabel/wmt22-comet-da")
                # data = {"src": [source_text], "mt": [hypothesis], "ref": [reference]}
                # comet_score = comet_model.predict(data, batch_size=1, gpus=0)[0]

                # Временная заглушка, если нет подключения к COMET в этом скрипте
                # ЗАМЕНИТЕ НА РЕАЛЬНЫЙ ВЫЗОВ ВАШЕЙ МОДЕЛИ COMET
                assessor_dummy = QualityAssessor()
                # Предполагаем, что в QualityAssessor есть метод get_comet или аналогичный
                # Если нет, нужно реализовать вызов здесь
                comet_score = assessor_dummy.calculate_comet(source_text, hypothesis, reference)
            except Exception as e:
                print(f"COMET calculation error: {e}")
                comet_score = 0.0

            # B. Безэталонные метрики (ИСПРАВЛЕННЫЕ)

            # 1. Semantic Similarity (BERTScore): Source (EN) vs Hypothesis (RU)
            # Мультиязычная модель сама разберется с векторами
            semantic_score = calculate_bert_score_multiling(source_text, hypothesis)

            # 2. Round-trip Consistency (chrF): Source (EN) vs Back Translation (EN)
            roundtrip_score = calculate_chrf(source_text, back_translation)

            # 3. NER Consistency: Entities(Source EN) vs Entities(Hypothesis RU)
            ents_src = extract_entities_en(source_text)
            ents_hyp = extract_entities_ru(hypothesis)
            ner_score = calculate_jaccard(ents_src, ents_hyp)

            # 4. Perplexity (GPT-2): Hypothesis (RU)?
            # Стоп. GPT-2 обучена на английском. Она не оценит русский текст.
            # В задании сказано: "Оценка гладкости текста...".
            # Если перевод делается НА РУССКИЙ, то нужна русская языковая модель для перплексии.
            # Например, ruGPT-3 или какая-то русская трансформер модель.
            # ИЛИ, если мы оцениваем английский перевод (если проект EN->RU, то гипотеза RU).
            # Тогда GPT-2 не подойдет. Нужно использовать русскую модель.
            # Давайте используем ruGPT-3 или просто другую доступную русскую LM.
            # Для простоты эксперимента, если нет русской LM, можно использовать mGPT или указать, что это ограничение.
            # Но чтобы было правильно: заменим GPT-2 на модель, понимающую русский, например 'sberbank-ai/ruGPT-3xl_8192' (тяжелая)
            # или 'cointegrated/rubert-tiny2' для перплексии.
            # Для скорости и доступности возьмем 'cointegrated/rubert-tiny2' или аналогичную маленькую русскую модель.

            # ПЕРЕОПРЕДЕЛЕНИЕ МОДЕЛИ ДЛЯ PERPLEXITY (РУССКАЯ)
            global ppl_model, ppl_tokenizer
            # Перезагружаем под русский, если еще не загружена (лучше сделать это один раз в начале)
            # Но в рамках этого скрипта сделаем проверку.
            # В реальном коде лучше вынести загрузку ru_model отдельно.

            try:
                # Используем небольшую русскую модель для перплексии
                # Если она еще не загружена в переменную выше (там был gpt2), надо заменить
                # Чтобы не ломать код выше, создадим локальную логику или заменим глобально.
                # Заменим глобально один раз при первом вызове, если модель была английской
                if not hasattr(run_experiment, 'ru_ppl_loaded'):
                    print("Loading Russian LM for Perplexity (ruGPT-3 tiny or similar)...")
                    # Используем distilrubert или что-то легкое, так как ruGPT-3 тяжелый
                    # Вариант: 'sergeyzh/rubert-tiny-tor' или просто считаем через ruBert
                    # Для чистоты эксперимента возьмем 'ai-forever/ruGPT-3-small' если есть доступ, иначе эвристику с предупреждением.
                    # НО пользователь просил GPT-2. Если текст РУССКИЙ, GPT-2 бесполезен.
                    # Скорее всего, в проекте перевод С РУССКОГО НА АНГЛИЙСКИЙ?
                    # Проверим датасет: "английским текстом разделённым (|||)". Обычно src ||| ref.
                    # Если src = EN, ref = RU. Значит перевод EN -> RU. Гипотеза = RU.
                    # Тогда Perplexity должна считаться на русском.
                    # Загрузим русскую модель.
                    ru_ppl_tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
                    ru_ppl_model = AutoModelForCausalLM.from_pretrained("cointegrated/rubert-tiny2").to(DEVICE)
                    ru_ppl_tokenizer.pad_token = ru_ppl_tokenizer.eos_token
                    run_experiment.ru_ppl_tokenizer = ru_ppl_tokenizer
                    run_experiment.ru_ppl_model = ru_ppl_model
                    run_experiment.ru_ppl_loaded = True

                tok = run_experiment.ru_ppl_tokenizer
                mod = run_experiment.ru_ppl_model

                encodings = tok(hypothesis, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
                with torch.no_grad():
                    outputs = mod(**encodings, labels=encodings.input_ids)
                    loss = outputs.loss
                perplexity_score_raw = float(torch.exp(loss))
                # Нормализуем перплексию в диапазон 0-1 (инвертируем, т.к. меньше перплексия = лучше)
                # Эвристика: 1 / (1 + log(perplexity)) или просто ограничим
                perplexity_score = 1.0 / (1.0 + np.log(perplexity_score_raw + 1e-5))

            except Exception as e:
                print(f"Ru Perplexity error: {e}")
                perplexity_score = 0.0

            # 5. Aggregate Score (Без эталона)
            # Веса из вашего отчета: Semantic 0.3, RoundTrip 0.4, NER 0.15, Perplexity 0.15
            aggregate_no_ref = (
                    semantic_score * 0.30 +
                    roundtrip_score * 0.40 +
                    ner_score * 0.15 +
                    perplexity_score * 0.15
            )

            results.append({
                "sample_id": idx,
                "temperature": temp,
                "provider_config": str(provider_config), # Или имя
                "source_text": source_text,
                "translation": hypothesis,
                "reference": reference,
                "back_translation": back_translation,
                "comet_score": comet_score,
                "semantic_score": semantic_score,
                "roundtrip_chrf": roundtrip_score,
                "ner_consistency": ner_score,
                "perplexity_score": perplexity_score, # Нормализованный
                "aggregate_no_ref": aggregate_no_ref
            })

    # Сохранение результатов
    df = pd.DataFrame(results)

    os.makedirs(os.path.dirname(OUTPUT_EXCEL), exist_ok=True)

    # Сохраняем в Excel
    df.to_excel(OUTPUT_EXCEL, index=False)
    print(f"Results saved to {OUTPUT_EXCEL}")

    # Сохраняем в CSV для удобства
    df.to_csv(OUTPUT_CSV, index=False, sep='\t')
    print(f"Results saved to {OUTPUT_CSV}")

    # Краткий анализ корреляции прямо в консоли
    if len(df) > 1:
        corr = df['comet_score'].corr(df['aggregate_no_ref'])
        print(f"\n=== CORRELATION RESULT ===")
        print(f"Pearson Correlation between COMET and Aggregate No-Ref: {corr:.4f}")
        print("==========================")

if __name__ == "__main__":
    run_experiment()