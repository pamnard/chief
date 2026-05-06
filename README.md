# chief

Проект **chief** — [github.com/pamnard/chief](https://github.com/pamnard/chief). Исходный код и публичные материалы в корне и в отслеживаемых git каталогах; приватные заметки и черновики не входят в репозиторий (см. `.gitignore`).

После установки (`pip install -e .`): `chief run пример задачи` (планировщик по умолчанию **fake**, без сети). Для LLM: реализация `chief.llm.HttpChatCompletionsBrain` (wire-формат Chat Completions; переменные `CHIEF_LLM_BASE_URL`, `CHIEF_LLM_MODEL`, при необходимости `CHIEF_LLM_API_KEY`), затем `chief run --brain llm …`. Локальный **Ollama** в совместимом режиме: база `http://127.0.0.1:11434/v1`.

## Лицензия

Распространение — на условиях [MIT License](LICENSE).
