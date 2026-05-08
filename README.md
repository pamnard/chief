# chief

[github.com/pamnard/chief](https://github.com/pamnard/chief)

Эпизодный цикл агента (**Perception → Reason → Act → Learn**): один CLI-эпизод, LLM через **httpx**, лог тиков в JSONL под XDG state. **Python ≥ 3.11.**

## Установка

```bash
pip install -e .
```

Для тестов и проверки стиля:

```bash
pip install -e ".[dev]"
pytest
```

## Конфигурация

- **Главный TOML:** в пакете зашиты дефолты (`chief/config/defaults.toml`). Поверх: **`$XDG_CONFIG_HOME/chief/chief.toml`**, опционально **`CHIEF_CONFIG`** / **`CHIEF_CONFIG_FILE`**, затем **`CHIEF_*`** для отдельных полей (см. `src/chief/config/loader.py`).
- **Реестр провайдеров (JSON):** в пакете `chief/config/defaults.providers.json`; поверх — **`$XDG_CONFIG_HOME/chief/providers.json`** (те же `id` заменяют или дополняют записи). Альтернативный путь: **`CHIEF_PROVIDERS_FILE`**. Формат: объект с массивом **`providers`**, каждый элемент — запись с полями **`id`**, **`kind`**, и параметрами эндпоинта.
- **Каталог моделей (JSON):** `chief/config/defaults.models.json` + **`$XDG_CONFIG_HOME/chief/models.json`** или **`CHIEF_MODELS_FILE`**; массив **`models`**. Связь с провайдером через **`provider_id`**; выбор записи для эпизода — **`[chief].default_model`** / **`CHIEF_DEFAULT_MODEL`** (см. ниже).
- **`kind`:** `custom_llm` (ваш шлюз с wire OpenAI Chat Completions), `openai` (вендор OpenAI), `anthropic`, `gemini`.
- **`enabled`:** у записи в JSON можно задать `"enabled": false` — провайдер не участвует в выборе и в probe; по умолчанию `true`.
- **Дефолтный планировщик:** в пакетных дефолтах **`[chief].default_provider = "custom_llm"`** (локальный Ollama из бандла). Для прогонов **без сети** задайте **`fake`** в `chief.toml`, **`CHIEF_DEFAULT_PROVIDER=fake`**, или запускайте **`pytest`** (в тестах это выставляется по умолчанию). Переопределение: **`CHIEF_DEFAULT_PROVIDER`**.
- **Readiness (фаза 2):** перед `chief run` с LLM-провайдером и при старте `chief serve` (если дефолтный провайдер не `fake`) выполняется короткий **probe** к API. Плейсхолдер ключа `REPLACE_ME_OR_USE_ENV` у вендоров считается неготовым конфигом. Отключить сетевые проверки (тесты, CI): **`CHIEF_SKIP_LLM_PROBE=1`**. Таймаут одного probe: **`CHIEF_LLM_PROBE_TIMEOUT`** (секунды, по умолчанию 5).

## Настройка провайдеров

### Где лежит реестр

1. В пакете уже есть **`defaults.providers.json`** (базовые записи).
2. Создайте каталог **`$XDG_CONFIG_HOME/chief/`** (обычно `~/.config/chief/`).
3. Файл **`providers.json`** в этом каталоге **накладывается поверх**: записи с тем же **`id`** заменяют бандл, новые **`id`** добавляются.
4. Либо укажите другой файл: переменная **`CHIEF_PROVIDERS_FILE`** (абсолютный или относительный путь к JSON).

Формат корня файла: объект с ключом **`providers`** — массив объектов-записей.

### Поля записи

Общие поля:

| Поле          | Описание                                                                                          |
| ------------- | ------------------------------------------------------------------------------------------------- |
| **`id`**      | Имя для CLI/IPC (`chief run --provider <id>`). Латиница, цифры, `_`, `-`.                         |
| **`kind`**    | Тип транспорта: см. таблицу ниже.                                                                 |
| **`enabled`** | Необязательно; по умолчанию `true`. При `false` запись нельзя выбрать и она не участвует в probe. |

Поля по **`kind`**:

| `kind`           | Обязательные поля                                           | Примечания                                                                                                                                                      |
| ---------------- | ----------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`custom_llm`** | `base_url`, `model`                                         | База в стиле OpenAI v1, например `http://127.0.0.1:11434/v1` (Ollama). `api_key` можно оставить пустым. Опционально: `timeout_seconds`.                         |
| **`openai`**     | `model`                                                     | `base_url` можно опустить — подставится официальный `https://api.openai.com/v1`. Нужен **реальный** `api_key` (не плейсхолдер). Опционально: `timeout_seconds`. |
| **`anthropic`**  | `base_url`, `api_version`, `api_key`, `model`, `max_tokens` | Типично `base_url`: `https://api.anthropic.com/v1`, `api_version`: например `2023-06-01`.                                                                       |
| **`gemini`**     | `base_url`, `api_key`, `model`                              | Типично `base_url`: `https://generativelanguage.googleapis.com/v1beta`.                                                                                         |

Идентификаторы **`custom_llm`**, **`openai`**, **`anthropic`**, **`gemini`** в бандле совпадают с внутренними «каноническими» слайсами рантайма; можно завести **дополнительные** строки с тем же `kind`, но **другим** `id` (например второй шлюз).

### Каталог моделей (фаза 3, первая итерация)

Отдельно от провайдеров: **`defaults.models.json`** в пакете (примеры записей) и пользовательский **`$XDG_CONFIG_HOME/chief/models.json`**, merge по полю **`id`** (как у провайдеров). Альтернатива: **`CHIEF_MODELS_FILE`**.

Каждая запись в массиве **`models`**:

| Поле                 | Описание                                                                                                                                                                   |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`id`**             | Логическое имя модели (для TOML / будущей маршрутизации).                                                                                                                  |
| **`provider_id`**    | Ссылка на **`id`** записи в реестре **провайдеров** (должен существовать).                                                                                                 |
| **`model`**          | Строка модели для API (для Chat Completions — поле `model` в теле запроса).                                                                                                |
| **`json_mode`**      | Необязательно, по умолчанию `false`. Для провайдеров **`custom_llm`** и **`openai`** включает OpenAI-совместимый `response_format: json_object`, если сервер поддерживает. |
| **`context_tokens`** | Необязательно; целое число, зарезервировано под лимиты/маршрутизацию.                                                                                                      |
| **`supports_tools`** | Необязательно, по умолчанию `false`; зарезервировано.                                                                                                                      |
| **`technical`**      | Необязательно, по умолчанию `false`. Если `true`, запись подходит для **вспомогательных** LLM-вызовов (сжатие контекста и т.д.); основной планер обычно с `false`.            |

В **`chief.toml`**: **`[chief].default_model`** — **`id`** из каталога. Тогда для провайдера, на который указывает эта запись (`provider_id`), при вызове Chat Completions подставляются **`model`** и **`json_mode`** из каталога; иначе используются поля из реестра провайдера. Переменная окружения: **`CHIEF_DEFAULT_MODEL`**. Если **`default_model`** не задан — поведение как раньше (модель только из провайдера, без JSON mode).

### Выбор провайдера по умолчанию

В **`chief.toml`** в секции **`[chief]`** задайте **`default_provider`** — строка **`fake`** (без сети; удобно для тестов) или **`id`** из реестра. В пакетных дефолтах по умолчанию **`custom_llm`**. Переопределение через **`CHIEF_DEFAULT_PROVIDER`**.

### Пример: локальный Ollama

Файл **`~/.config/chief/providers.json`**:

```json
{
  "providers": [
    {
      "id": "custom_llm",
      "kind": "custom_llm",
      "base_url": "http://127.0.0.1:11434/v1",
      "model": "gemma2:2b",
      "api_key": "",
      "timeout_seconds": 120
    }
  ]
}
```

В **`~/.config/chief/chief.toml`**:

```toml
[chief]
default_provider = "custom_llm"
```

### Быстрый старт (интерактив)

При **TTY**:

```bash
chief setup providers
```

Команда создаёт/обновляет **`chief.toml`** и **`providers.json`** для канонического **`custom_llm`** (URL и модель задаются вопросами; пустой ответ — значение по умолчанию в подсказке). Без TTY используйте ручное редактирование файлов ниже.

Проверка: `chief run --provider custom_llm "короткий тест"`.

### Пример: OpenAI

В **`providers.json`** для записи с `"id": "openai"` замените **`api_key`** на рабочий ключ (или заведите новую запись с другим `id` и `kind: "openai"`). Строка **`REPLACE_ME_OR_USE_ENV`** из бандла для probe считается **незаполненной** — эпизод с таким ключом не стартует, пока ключ не задан.

### Секреты

Ключи лучше не коммитить: храните их в **`providers.json`** только на машине пользователя (каталог XDG), либо подставляйте через свой процесс (редактируете файл вручную / внешний секрет-хранилище). В логах и сообщениях об ошибках ключи не выводятся.

## CLI

Справка по встроенным флагам:

```bash
chief --help
chief run --help
chief serve --help
chief chat --help
chief setup --help
```

### `chief setup` — мастер конфигурации (TTY)

```text
chief setup providers
```

Интерактивно записывает **`chief.toml`** и **`providers.json`** под XDG (см. выше). Без интерактивного терминала завершается с кодом **2** и подсказкой править файлы вручную.

### `chief run` — один эпизод в процессе

Синтаксис:

```text
chief run [ТЕКСТ ЗАДАЧИ …] [--provider ID] [--max-cycles N] [--json]
```

| Опция          | Значение по умолчанию             | Описание                                                                                                     |
| -------------- | --------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| `task`         | (пусто)                           | Все аргументы после `run` склеиваются через пробел в строку задачи для эпизода.                              |
| `--provider`   | из `[chief].default_provider`     | `fake` или **id** из реестра провайдеров (`custom_llm`, `openai`, …).                                        |
| `--max-cycles` | из конфига `[episode].max_cycles` | Жёсткий потолок циклов перепланирования (replan).                                                            |
| `--json`       | выкл.                             | Вместо человекочитаемого вывода — один JSON-объект: `id`, `status`, `artifact`, `ticks`, `task`, `provider`. |

Коды выхода: **0** если финальный статус эпизода `completed`, иначе **1**.

Примеры:

```bash
chief run hello world
chief run --provider fake "echo this"
chief run --provider custom_llm разбери задачу --max-cycles 8
chief run --provider openai кратко ответь --json
```

### `chief serve` — фоновый сервер (Unix socket, NDJSON)

Синтаксис:

```text
chief serve [--provider ID] [--socket ПУТЬ_К_СОКЕТУ]
```

| Опция        | Значение по умолчанию                | Описание                                                                                                                                     |
| ------------ | ------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `--provider` | из `[chief].default_provider`        | Планировщик по умолчанию для клиентов с `target=orchestrator` в протоколе v1.                                                                |
| `--socket`   | из `RuntimeConfig.serve_socket_path` | Переопределить путь Unix-сокета; иначе `[serve].socket_path` в TOML, **`CHIEF_SERVE_SOCKET`**, либо каталог XDG runtime и файл `chief.sock`. |

Процесс слушает сокет до **SIGINT** / **SIGTERM**. Рядом с сокетом создаётся **`chief-serve.pid`**.

Коды выхода: **0** при нормальном завершении, **1** при ошибке ОС (например нельзя создать сокет).

Пример (два терминала):

```bash
# терминал A
chief serve --provider fake

# терминал B (тот же сокет по умолчанию или тот же --socket)
chief chat
```

### `chief chat` — REPL: stdin → сервер

Синтаксис:

```text
chief chat [--socket ПУТЬ] [--session ID] [--target {orchestrator|subagent}] [--provider ID] [--verbose]
```

| Опция        | Значение по умолчанию       | Описание                                                                                                                                     |
| ------------ | --------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `--socket`   | `runtime.serve_socket_path` | Сокет запущенного `chief serve`.                                                                                                             |
| `--session`  | `cli`                       | Поле `session_id` в каждой NDJSON-строке запроса.                                                                                            |
| `--target`   | `orchestrator`              | `orchestrator` — провайдер из `--provider` сервера; `subagent` — провайдер из **`--provider` клиента** (обязателен).                         |
| `--provider` | не задаётся                 | Для **`--target subagent`** обязателен: **`fake`** или id из реестра. Для **`orchestrator`** задавать нельзя (клиент шлёт `provider: null`). |
| `--verbose`  | выкл.                       | Перед текстом ответа вывести строку `status=… id=… ticks=…` (дебаг эпизода).                                                                 |

Протокол v1: в JSON запроса поле **`provider`** (не `brain`). Каждая непустая строка **stdin** — отдельный запрос; сервер отвечает одной строкой JSON. В **stdout** по умолчанию печатается только **`artifact`**; с **`--verbose`** — сначала строка эпизода, затем **`artifact`**.

Коды выхода: **0** успех, **1** ошибка сокета / соединения / `RuntimeError`, **2** если указан **`--target subagent`** без **`--provider`**.

Примеры:

```bash
chief chat
chief chat --verbose
chief chat --session mysession --target orchestrator
chief chat --socket /run/user/1000/chief/chief.sock --target subagent --provider fake
```

## Лицензия

[MIT License](LICENSE).
