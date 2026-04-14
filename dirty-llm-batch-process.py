import os
import sys
import time
from openai import OpenAI
from tqdm import tqdm

# --- НАСТРОЙКИ ---
client = OpenAI(base_url="http://localhost:5000/v1", api_key="not-needed")

SYSTEM_PROMPT = r"""
You are an expert summarization assistant specializing in creating detailed, citation-backed summaries in russian.\nYour task:
Create a comprehensive summary of the provided video transcript or article text. The summary should be in russian and preserve all important insights while being significantly shorter than the original.\nCore requirements:\nStructure your output with clear section headings that reflect the natural flow of the content (introduction, main topics, conclusion). Write in continuous paragraphs under each heading — avoid bullet points, lists, tables, or emoji.\nFor every key claim or insight, include a short direct quote from the original text in quotation marks translated to the Russian: "exact words from source". When working with video transcripts that include timestamps, always cite them after quotes in square brackets: "quote text" [12:34]. For text without timestamps, cite section or page numbers if available.\nCompress the content to roughly 20-30% of original length while preserving 90-100% of important insights, counterintuitive points, non-obvious details, and technical definitions. Remove filler words, repetitions, digressions, sponsor messages, and casual banter.\nBegin each major section with a timestamp range if working with video: [00:00-05:30]. This allows readers to jump to specific parts of the original material.\nWrite in clear, technical language suitable for someone who wants to understand the material deeply without watching or reading the entire source. Maintain the author's original intent and terminology. When complex terms appear, explain them inline using the context from the source material.\nYour summary should serve as detailed study notes that can replace the original for review purposes, not as a high-level vague overview.\n\n
"""

INPUT_DIR = "./INPUT_DIR"
OUTPUT_DIR = "./OUTPUT_DIR"
MODEL_NAME = "local-model"
PRINT_INTERVAL = 10.0  # Интервал обновления консоли (сек)

def exit_with_pause(message, is_error=False):
    """Выводит сообщение и ждет нажатия Enter перед выходом."""
    prefix = "!!! ОШИБКА: " if is_error else ""
    print(f"\n{prefix}{message}")
    input("\nНажмите Enter, чтобы закрыть это окно...")
    sys.exit()

def process_files():
    # 1. Проверка входной папки
    if not os.path.exists(INPUT_DIR):
        exit_with_pause(f"Папка '{INPUT_DIR}' не найдена. Создайте её и положите файлы.", is_error=True)

    all_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".txt")]
    if not all_files:
        exit_with_pause(f"В папке '{INPUT_DIR}' нет текстовых файлов (.txt).", is_error=True)

    # 2. Проверка выходной папки и пересечений
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    existing_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".txt")]
    # Ищем файлы, которые уже есть в обеих папках
    intersection = list(set(all_files) & set(existing_files))

    # 3. Решение о запуске или вопросе пользователю
    if not intersection:
        # Если совпадений нет, просто берем все файлы
        files_to_process = all_files
        print(f">> Совпадений не найдено. Начинаю обработку всех файлов ({len(all_files)} шт.).")
    else:
        print(f"\nНайдено совпадений: {len(intersection)} шт. (уже есть в папке результата)")
        choice = input("[S]kip (пропустить готовые), [O]verwrite (перезаписать всё), [A]bort (отмена): ").lower()
        
        if choice in ['s', 'ы', 'c', 'с']:
            files_to_process = [f for f in all_files if f not in existing_files]
            print(">> Пропускаем уже обработанные.")
        elif choice in ['o', 'щ', 'j', 'о', 'y', 'н']:
            files_to_process = all_files
            print(">> Перезаписываем все.")
        else:
            exit_with_pause("Операция отменена пользователем.")

    if not files_to_process:
        exit_with_pause("Нет новых файлов для обработки.")

    # --- ОСНОВНОЙ ЦИКЛ ОБРАБОТКИ ---
    total_p_tokens = 0
    total_c_tokens = 0
    
    pbar = tqdm(files_to_process, desc="Прогресс", unit="file", mininterval=1e6)

    for filename in pbar:
        full_response_text = ""
        curr_file_tokens = 0
        last_ui_time = time.time()
        tokens_at_last_update = 0
        
        pbar.set_description(f"Файл: {filename[:20]}")
        pbar.refresh()

        try:
            with open(os.path.join(INPUT_DIR, filename), 'r', encoding='utf-8') as f:
                content = f.read()

            stream = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": content}
                ],
                stream=True,
                stream_options={"include_usage": True}
            )

            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    full_response_text += chunk.choices[0].delta.content
                    curr_file_tokens += 1 

                if chunk.usage:
                    curr_file_tokens = chunk.usage.completion_tokens
                    total_p_tokens += chunk.usage.prompt_tokens
                    total_c_tokens += chunk.usage.completion_tokens

                # Троттлинг вывода (раз в 10 секунд)
                now = time.time()
                delta_t = now - last_ui_time
                if delta_t >= PRINT_INTERVAL:
                    new_tokens = curr_file_tokens - tokens_at_last_update
                    speed = new_tokens / delta_t if delta_t > 0 else 0
                    
                    pbar.set_postfix({
                        'cur_tok': curr_file_tokens,
                        't/s': f"{speed:.1f}",
                        'total': total_p_tokens + total_c_tokens + curr_file_tokens
                    })
                    
                    preview = full_response_text[-120:].replace("\n", " ")
                    pbar.write(f"[{filename}] {speed:.1f} t/s | Превью: \s{preview}\s")
                    pbar.refresh()
                    
                    last_ui_time = now
                    tokens_at_last_update = curr_file_tokens

            # Сохранение
            with open(os.path.join(OUTPUT_DIR, filename), 'w', encoding='utf-8') as f:
                f.write(full_response_text)
            
            pbar.refresh()

        except Exception as e:
            pbar.write(f"!!! Ошибка в {filename}: {e}")

    # Финальный аккорд
    print(f"\n{'='*50}\nГОТОВО! Итого токенов: {total_p_tokens + total_c_tokens}\n{'='*50}")
    exit_with_pause("Все задачи выполнены.")

if __name__ == "__main__":
    try:
        process_files()
    except KeyboardInterrupt:
        exit_with_pause("Программа прервана пользователем (Ctrl+C).")
    except Exception as e:
        exit_with_pause(f"Непредвиденная ошибка: {e}", is_error=True)
