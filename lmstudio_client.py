"""
Проект сравнения методов оценки качества перевода нейронных сетей
1. Оценка с эталоном (BLEU, ROUGE, BERTScore)
2. Оценка без эталона (Back-translation consistency)
"""

import requests
import json
import logging
from typing import Optional
from datetime import datetime

# Настройка логирования
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('lmstudio_debug.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class LMStudioClient:
    """Клиент для работы с локальной нейросетью через LM Studio"""

    def __init__(self, base_url: str = "http://localhost:1234"):
        self.base_url = base_url
        self.api_endpoint = f"{base_url}/api/v1/chat"
        logger.info(f"Инициализация клиента. API endpoint: {self.api_endpoint}")

    def translate(
            self,
            text: str,
            source_lang: str,
            target_lang: str,
            model: str = "gigachat3.1-10b-a1.8b"
    ) -> Optional[str]:
        """
        Перевод текста с одного языка на другой

        Args:
            text: Текст для перевода
            source_lang: Язык исходного текста
            target_lang: Язык перевода
            model: Название модели

        Returns:
            Переведенный текст или None при ошибке
        """
        system_instruction = (
            f"You are a professional translator. Translate the following text from {source_lang} to {target_lang}. "
            "Return ONLY the translated text, without any explanations."
        )

        payload = {
            "model": model,
            "system_prompt": system_instruction,
            "input": text
        }

        logger.debug(f"Отправка запроса на перевод:")
        logger.debug(f"  URL: {self.api_endpoint}")
        logger.debug(f"  Модель: {model}")
        logger.debug(f"  Языки: {source_lang} -> {target_lang}")
        logger.debug(f"  Входной текст (первые 100 символов): {text[:100]}...")
        logger.debug(f"  Payload: {json.dumps(payload, ensure_ascii=False)[:200]}...")

        try:
            logger.debug("Выполнение POST запроса...")
            start_time = datetime.now()

            response = requests.post(
                self.api_endpoint,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=60
            )

            elapsed = (datetime.now() - start_time).total_seconds()
            logger.debug(f"Получен ответ за {elapsed:.2f} сек. Статус код: {response.status_code}")
            logger.debug(f"Заголовки ответа: {dict(response.headers)}")

            response.raise_for_status()

            logger.debug(f"Тело ответа: {response.text[:500]}...")

            result = response.json()
            logger.debug(f"Распарсенный JSON: {result}")

            # LM Studio возвращает ответ в формате:
            # {"output": [{"type": "message", "content": "..."}], ...}
            translation = ""
            if "output" in result and isinstance(result["output"], list) and len(result["output"]) > 0:
                translation = result["output"][0].get("content", "").strip()
            elif "response" in result:
                translation = result.get("response", "").strip()
            else:
                logger.warning("Неизвестный формат ответа")
                logger.warning(f"Полный ответ: {result}")

            if not translation:
                logger.warning("Ответ получен, но перевод пустой")
                logger.warning(f"Полный ответ: {result}")
                return None

            logger.info(f"Перевод успешно получен (длина: {len(translation)} символов)")
            return translation

        except requests.exceptions.ConnectionError as e:
            logger.error(f"Ошибка подключения: {e}")
            logger.error("Возможные причины:")
            logger.error("  1. LM Studio не запущен")
            logger.error("  2. Неправильный адрес сервера (по умолчанию http://localhost:1234)")
            logger.error("  3. Сервер не слушает указанный порт")
            logger.error("Проверьте, что LM Studio запущен и локальный сервер активен!")
            return None

        except requests.exceptions.Timeout as e:
            logger.error(f"Таймаут запроса: {e}")
            logger.error("Сервер не ответил в течение 60 секунд")
            return None

        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP ошибка: {e}")
            logger.error(f"Статус код: {response.status_code}")
            logger.error(f"Тело ответа: {response.text}")
            return None

        except json.JSONDecodeError as e:
            logger.error(f"Ошибка парсинга JSON ответа: {e}")
            logger.error(f"Полученный текст: {response.text}")
            return None

        except Exception as e:
            logger.error(f"Неожиданная ошибка при переводе: {e}", exc_info=True)
            return None

    def back_translate(
            self,
            text: str,
            source_lang: str,
            target_lang: str,
            model: str = "gigachat3.1-10b-a1.8b"
    ) -> Optional[str]:
        """
        Обратный перевод (для оценки без эталона)

        Args:
            text: Текст для обратного перевода
            source_lang: Исходный язык (оригинала)
            target_lang: Язык текста (куда перевели)
            model: Название модели

        Returns:
            Текст после обратного перевода
        """
        actual_source = source_lang if source_lang != 'auto' else 'en'

        back_system_instruction = (
            f"You are a professional translator. Translate the following text from {target_lang} to {actual_source}. "
            "Return ONLY the translated text, without any explanations."
        )

        payload = {
            "model": model,
            "system_prompt": back_system_instruction,
            "input": text
        }

        logger.debug(f"Отправка запроса на обратный перевод:")
        logger.debug(f"  URL: {self.api_endpoint}")
        logger.debug(f"  Модель: {model}")
        logger.debug(f"  Языки: {target_lang} -> {actual_source}")
        logger.debug(f"  Входной текст (первые 100 символов): {text[:100]}...")

        try:
            logger.debug("Выполнение POST запроса...")
            start_time = datetime.now()

            response = requests.post(
                self.api_endpoint,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=60
            )

            elapsed = (datetime.now() - start_time).total_seconds()
            logger.debug(f"Получен ответ за {elapsed:.2f} сек. Статус код: {response.status_code}")

            response.raise_for_status()

            logger.debug(f"Тело ответа: {response.text[:500]}...")

            result = response.json()
            logger.debug(f"Распарсенный JSON: {result}")

            # LM Studio возвращает ответ в формате:
            # {"output": [{"type": "message", "content": "..."}], ...}
            translation = ""
            if "output" in result and isinstance(result["output"], list) and len(result["output"]) > 0:
                translation = result["output"][0].get("content", "").strip()
            elif "response" in result:
                translation = result.get("response", "").strip()
            else:
                logger.warning("Неизвестный формат ответа")
                logger.warning(f"Полный ответ: {result}")

            if not translation:
                logger.warning("Ответ получен, но перевод пустой")
                logger.warning(f"Полный ответ: {result}")
                return None

            logger.info(f"Обратный перевод успешно получен (длина: {len(translation)} символов)")
            return translation

        except requests.exceptions.ConnectionError as e:
            logger.error(f"Ошибка подключения при обратном переводе: {e}")
            logger.error("Проверьте, что LM Studio запущен и локальный сервер активен!")
            return None

        except requests.exceptions.Timeout as e:
            logger.error(f"Таймаут запроса при обратном переводе: {e}")
            return None

        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP ошибка при обратном переводе: {e}")
            logger.error(f"Статус код: {response.status_code}")
            logger.error(f"Тело ответа: {response.text}")
            return None

        except json.JSONDecodeError as e:
            logger.error(f"Ошибка парсинга JSON ответа при обратном переводе: {e}")
            logger.error(f"Полученный текст: {response.text}")
            return None

        except Exception as e:
            logger.error(f"Неожиданная ошибка при обратном переводе: {e}", exc_info=True)
            return None