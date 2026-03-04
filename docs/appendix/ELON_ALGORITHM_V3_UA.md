# АЛГОРИТМ ІЛОНА — ФІНАЛЬНА ВЕРСІЯ v3
## Калібровано до реального коду: step_inplace / sero_step / StepOutputs / clamp_V_inplace

---

> **Фрактальний принцип**: кожен промпт — одна трансформація над одним об'єктом.
> **Рекурсивний принцип**: вивід Промпту N є точним незмінним входом Промпту N+1.
> **Циклічний принцип**: якщо Промпт 5 падає — він вказує точно в який Промпт повернутись.
> **Заборона**: Промпт N+1 не запускається якщо Промпт N не має верифікованого виводу.

---

## ПРОМПТ 1 — REQUIREMENTS KILL AUDIT
### Єдине питання: "Без цього step_inplace() або sero_step() видадуть неправильний результат?"

```text
РЕЖИМ: Прокурор. Не адвокат. Не порадник. Виконавець знищення.

═══════════════════════════════════════════════════════════
ФІЗИЧНА РЕАЛЬНІСТЬ СИСТЕМИ (прочитай — це не опис, це закон)
═══════════════════════════════════════════════════════════

Єдиний hot path neuromorphic (реальний код):

  step_inplace(view, params, dt, rng) → StepOutputs
    де StepOutputs = {
      phi:              float               ← фаза [0, 2π]
      alert:            float               ← рівень загрози
      global_state:     NodeState           ← REST/ACTIVE/ADAPTIVE/CRITICAL
      nan_inf_detected: bool                ← контракт NaN/Inf
      v_p99_pre_reset:  float               ← вхід у sero_step()
    }

  sero_step(cfg, st, v_p99_mv, nan_inf) → SEROState
    де SEROState = {
      multiplier: float ∈ [min_mult, max_mult]  ← це і є weight_scale
    }

  Якщо nan_inf=True → sero_step() зменшує multiplier (вбудований захист).
  Якщо V виходить за [-100, 50] → clamp_V_inplace() → відновлення.

SERO M0 (окремий прототип sero_m0.py):
  e = x - x_hat
  S = clamp(S_prev + γ·(‖e‖_s - S_prev), 0, S_max)
  T = T_min + (T_0 - T_min)·exp(-α·S)
  gate = sigmoid(log_odds) > θ
  Вивід: T ∈ [T_min, T_0], S ∈ [0, S_max], escalation (bool)

═══════════════════════════════════════════════════════════
ВХІД
═══════════════════════════════════════════════════════════
[вставити: повне дерево файлів + requirements.txt + список env-змінних]

═══════════════════════════════════════════════════════════
КРОК 1. ДОПИТ КОЖНОГО КОМПОНЕНТА
═══════════════════════════════════════════════════════════

Для кожного модуля / класу / функції / поля / env-змінної / залежності / тесту:

  Q1. Це безпосередньо обслуговує step_inplace() або sero_step()?
      Якщо ні — підозрюваний.
  Q2. Яке конкретне поле StepOutputs або SEROState стане неправильним без цього?
      Якщо "жодне" — DUMB. KILL.
  Q3. Це тут тому що ФІЗИЧНО ЗЛАМАЄТЬСЯ, чи тому що "може знадобитись"?

ВБИВАЄМО БЕЗ ДОКАЗУ ПРОТИЛЕЖНОГО (реальні назви з коду):
  ─ Kafka / aiokafka          → не в step_inplace(), не в sero_step()
  ─ boto3 / S3                → не в жодному kernel
  ─ prometheus-client         → observability, не physics
  ─ FastAPI / uvicorn         → transport, не engine
  ─ igraph / cugraph          → modularity не входить в alert формулу
  ─ jax_kernel.py             → cpu_numpy є production backend
  ─ triton_kernel.py          → cpu_numpy є production backend
  ─ biochemical_capacity > 1  → якщо kernel використовує ≤2 речовини
  ─ ion_channels у domain.py  → якщо не входять в dV розрахунок
  ─ env dict pH/temperature   → якщо noise_scale — єдиний env param в kernel
  ─ per-node V_p50/p90        → StepOutputs має тільки v_p99_pre_reset
  ─ dependency-injector       → якщо container.py можна замінити двома рядками
  ─ lat_p50 у SERO M0         → severity_weight = 0.0, мертве поле
  ─ Claims C1/C5/C9-C14       → SPECULATED, не верифіковані, не в Eq.X
  ─ server.py / Dockerfile    → demo transport, не алгоритм
  ─ build_release_assets.py   → ZIP-пакування ≠ фізична необхідність
  ─ promptops/                → meta-tooling, не в hot path

═══════════════════════════════════════════════════════════
КРОК 2. ЄДИНА МЕТА
═══════════════════════════════════════════════════════════

Одне речення:
"Система існує щоб [дія] → StepOutputs{phi, alert, nan_inf_detected} + SEROState{multiplier}
 за умови V ∈ [-100,50], nan_inf=False, DETERMINISTIC=1 → біт-в-біт при RNG_SEED=42."

═══════════════════════════════════════════════════════════
КРОК 3. ВИВІД — ТІЛЬКИ ЦЕЙ ФОРМАТ
═══════════════════════════════════════════════════════════

ЄДИНА МЕТА: [одне речення]

ЗАЛИШИТИ (максимум 7 — кожен прив'язаний до поля StepOutputs або SEROState):
1. [компонент] — обслуговує [яке поле / яке рівняння]
...

ЗНИЩИТИ:
- [компонент] → ПРИЧИНА: [одне речення] | КЛАС: [Redundant/Premature/Fear-driven/Ego-driven]

ВЕРИФІКАЦІЯ ВИВОДУ (обов'язково після кожного пункту ЗАЛИШИТИ):
  → Яке поле StepOutputs або SEROState залежить від цього компонента?
  → Якщо відповідь порожня — компонент переміщується в ЗНИЩИТИ.

Заборонено: "можливо", "варто", "може знадобитись", "потенційно корисний".
```

---

## ПРОМПТ 2 — DELETE STRESS TEST
### Правило Ілона: якщо не повернув ≥10% — ти не видаляв. Повертайся в Промпт 1.

```text
РЕЖИМ: Знищення через доказ від противного. Презумпція смерті.

ВХІД: список ЗАЛИШИТИ з Промпту 1 (тільки він, нічого більше)

═══════════════════════════════════════════════════════════
ФІЗИЧНІ КРИТЕРІЇ СМЕРТІ
═══════════════════════════════════════════════════════════

KILL якщо хоча б один пункт TRUE:

  [ ] Можна замінити np.clip() / константою / одним assert-ом
  [ ] Не присутній у тілі step_inplace() або sero_step() — перевір реальний код
  [ ] Існує тільки для логів, метрик, "майбутнього API", observability
  [ ] Дублює clamp_v_mv() / clamp_weight() / safety floor T_min
  [ ] nan_inf_detected вже обробляє цей кейс — тоді окрема перевірка зайва

KEEP тільки якщо виконуються ОБА умови одночасно:
  → без цього конкретне поле StepOutputs або SEROState стає NaN/Inf або виходить за межі
  → І немає обхідного шляху через вже-існуючий clamp або nan_inf прапор

═══════════════════════════════════════════════════════════
АЛГОРИТМ ДЛЯ КОЖНОГО ЕЛЕМЕНТА
═══════════════════════════════════════════════════════════

  1. Знайди обхідний шлях без цього елемента.
  2. Обхідний шлях існує → DELETE.
  3. Система фізично ламається (не "гіршає" — саме ЛАМАЄТЬСЯ, конкретне поле) → KEEP.
  4. Сумнів → DELETE. Тест впаде — повернемо. Не впаде — правильно видалили.

═══════════════════════════════════════════════════════════
ВИВІД — ТІЛЬКИ ТАБЛИЦЯ
═══════════════════════════════════════════════════════════

| # | Компонент | D/K | Поле що ламається (або обхідний шлях якщо DELETE) |
|---|-----------|-----|---------------------------------------------------|

Залишилося: X компонентів
Повернуто назад: Y (Z%)

Якщо Z < 10%:
  ⚠ AUDIT FAILED.
  Причина: або Промпт 1 не видалив достатньо (повертайся в П1),
           або ти застосував KILL без реального пошуку обхідного шляху.
  Дія: повтори П1 з новим списком підозрюваних за замовчуванням.

Якщо Z ≥ 10%:
  ✓ Список підтверджено. Передаємо в Промпт 3.
```

---

## ПРОМПТ 3 — SIMPLIFY TO PHYSICS
### "The best part is no part" — якщо існує, доведи що не може бути масивом або константою.

```text
РЕЖИМ: Перший принцип. Складність — фізична субстанція яку знищуємо.

ВХІД: таблиця KEEP з Промпту 2 (тільки вона)

═══════════════════════════════════════════════════════════
РЕАЛЬНИЙ HOT PATH (це канон — спрощуй до цього, не від цього)
═══════════════════════════════════════════════════════════

neuromorphic (реальний код з cpu_numpy_kernel.py):

  # 1. Synaptic transmission (spike-gated + fatigue)
  due      = has_spiked[src] & (tick == last_spike_tick[src] + delay_steps)
  prob     = clip(0.95 * exp(-fatigue[due]), 0, 1)
  transmit = rng.random(prob.shape) < prob
  total_synaptic += bincount(dst[due][transmit],
                             weights=w[due][transmit] * weight_scale * (1 - fatigue),
                             minlength=N)

  # 2. Membrane dynamics (HH-simplified)
  gating = sigmoid((V + 40) / 10)
  i_total = 120*gating*(V-50) + 36*gating*(V+77) + 0.3*gating*(V+70)
  i_ext   = noise_scale * rng.standard_normal(N)
  dV      = (dt / c_m) * (-i_total + total_synaptic + i_ext)
  V      += dV  # тільки для not_refractory
  clamp_V_inplace(V, -100, 50)

  # 3. Alert
  alert = lambda*spike_rate + mu*phase_sync + rho*weight_scale

  # 4. SERO controller
  if nan_inf_detected or v_p99 > v_p99_target:
      multiplier *= exp(-gain_down * overshoot)
  elif v_p99 < v_p99_target - margin:
      multiplier = min(1.0, multiplier * (1 + gain_up))
  weight_scale = clamp(multiplier, min_mult, max_mult)

SERO M0 (реальний цикл з sero_m0.py):
  x_hat += alpha * (x - x_hat)              # EWMA
  e      = x - x_hat
  S      = clamp(S + gamma*(norm_s(e) - S), 0, S_max)
  T      = T_min + (T_0 - T_min)*exp(-alpha*S)
  log_odds = sum(w_i * evidence_conf * detector_i for i)
  gate   = sigmoid(log_odds) > theta

═══════════════════════════════════════════════════════════
ПРАВИЛА СПРОЩЕННЯ (застосуй до кожного KEEP елемента)
═══════════════════════════════════════════════════════════

→ Клас з одним методом і одним полем → def + return
→ Optional[List[float]] state → np.ndarray (EWMA estimate)
→ Pydantic у domain.py + validation.py → ONE SOURCE.
   domain.py: тільки структури даних без логіки.
   validation.py: єдине місце всіх clamp_*().
   Дублювання → KILL одне.
→ BiochemicalRegistry якщо N_chemicals ≤ 2 → dict literal або HARDCODE index
→ if/else з двома гілками де одна = "нічого не робити" → multiply by bool mask
→ dependency-injector container → якщо це просто create(engine) → два рядки
→ KernelParams dataclass → якщо всі поля константи для production run → CONSTANTS dict

═══════════════════════════════════════════════════════════
ВИВІД
═══════════════════════════════════════════════════════════

ДО → ПІСЛЯ (3–5 найжирніших перетворень):
[компонент]: [що було] → [що стало] | Видалено: [N рядків / N класів / N if-гілок]

ФІНАЛЬНЕ ЯДРО:
[мінімальний виконуваний код — тільки те що залишилось після спрощення]
[повинно читатись як рівняння, не як framework]

МЕТРИКА:
  Операцій у hot path neuromorphic: __ → __
  Операцій у SERO loop:             __ → __
  Рядків коду src/:                 __ → __
  Шарів абстракції між input і StepOutputs: __ → __

УМОВА ПРОВАЛУ:
  Якщо фінальне ядро > 60 рядків — Промпт 2 не завершено.
  Повернись в П2 і видали ще раз.
```

---

## ПРОМПТ 4 — CYCLE TIME COLLAPSE
### Один цикл "змінив → побачив результат" < 10 секунд. Все що довше — ворог.

```text
РЕЖИМ: Швидкість зворотного зв'язку — єдина метрика розробника.

ВХІД: фінальне ядро з Промпту 3

═══════════════════════════════════════════════════════════
РЕАЛЬНИЙ СТАН (що є в Makefile прямо зараз)
═══════════════════════════════════════════════════════════

verify: lint + type + test + perf   ← 4 кроки, вбивають швидкість
lint:   python -m flake8 src tests  ← ~5-15 сек
type:   python -m mypy src tests    ← ~20-60 сек
test:   python -m pytest -q         ← весь suite, ~30-120 сек
perf:   pytest tests/perf/... -k smoke ← окремий запуск

Реальні тести що вже існують і перевіряють фізичні контракти:
  tests/test_numerical_safety.py  ← NaN/Inf контракт (ПРЯМО В DEV LOOP)
  tests/test_determinism.py       ← детермінізм seed=42 (ПРЯМО В DEV LOOP)
  tests/claims/test_c3_safety_floor.py ← T ≥ T_min (ПРЯМО В DEV LOOP)
  tests/claims/test_c4_damping.py      ← S ≤ S_max (ПРЯМО В DEV LOOP)

Все інше — в pre-commit або CI.

═══════════════════════════════════════════════════════════
АУДИТ КОЖНОГО КРОКУ
═══════════════════════════════════════════════════════════

Для кожного: lint / mypy / pytest-full / perf / snapshot / release:
  → Час виконання на локальному ноутбуці (секунди)?
  → Чи виявляє порушення nan_inf_detected=True або детермінізм? Якщо ні → не в dev loop.
  → Де місце: dev (<10с) / pre-commit (<60с) / CI only / nowhere?

═══════════════════════════════════════════════════════════
ТРИ РІВНІ ЦИКЛУ
═══════════════════════════════════════════════════════════

DEV LOOP (< 10 секунд):
  Перевіряє тільки фізичні контракти — NaN/Inf + детермінізм + safety floor.
  Реальна команда:
    DETERMINISTIC=1 RNG_SEED=42 ENGINE_BACKEND=cpu_numpy SNAPSHOT_BACKEND=disabled \
    pytest tests/test_numerical_safety.py tests/test_determinism.py \
           tests/claims/test_c3_safety_floor.py tests/claims/test_c4_damping.py -q
  Очікуваний час: __ сек

PRE-COMMIT (< 60 секунд):
  + повна перевірка детермінізму (2 незалежні прогони)
  + SERO M0: python sero_m0.py → assert "7/7 PASS"
  + flake8 тільки src/ (не tests/)
  Команди: [список]
  Очікуваний час: __ сек

CI ONLY (без обмежень):
  mypy src tests
  pytest -q (повний suite)
  snapshot roundtrip
  build_release_assets.py

═══════════════════════════════════════════════════════════
ВИВІД
═══════════════════════════════════════════════════════════

Цільовий час dev loop: __ сек (ціль < 10)

Dev loop команда:
[одна команда або make target]

Pre-commit команди:
[список]

Видалено з dev loop:
- [що] → [скільки секунд економить] → [де тепер: pre-commit / CI / nowhere]

УМОВА ПРОВАЛУ:
  Якщо dev loop > 10 сек → Промпт 3 не завершено (є зайві залежності).
  Якщо в dev loop є mypy, flake8, Docker, kafka → Промпт 1 не завершено.
```

---

## ПРОМПТ 5 — AUTOMATE THE VERIFIED MINIMUM
### Один Makefile. ≤20 рядків. Від нуля до верифікованого стану.

```text
РЕЖИМ: Автоматизація тільки того що вижило після 4 кроків.

ВХІД: результати Промптів 1-4

═══════════════════════════════════════════════════════════
ПРАВИЛА (порушення = повернення в попередній промпт)
═══════════════════════════════════════════════════════════

  Makefile > 20 рядків → Промпт 3 не завершено. Повернись.
  smoke > 30 секунд    → Промпт 4 не завершено. Повернись.
  є Docker/Kafka/Ray   → Промпт 1 не завершено. Повернись.
  є --no-kafka у CLI   → помилка: це env vars, не CLI flags. Виправ.

═══════════════════════════════════════════════════════════
РЕАЛЬНІ ENV VARS (з config.py — не вигадуй нові)
═══════════════════════════════════════════════════════════

  DETERMINISTIC=1
  RNG_SEED=42
  ENGINE_BACKEND=cpu_numpy
  SNAPSHOT_BACKEND=disabled
  RAY_ENABLED=0
  KAFKA_BROKERS=""
  LOG_LEVEL=WARNING

═══════════════════════════════════════════════════════════
ЗАВДАННЯ: ОДИН MAKEFILE
═══════════════════════════════════════════════════════════

install:
  pip install numpy scipy pydantic pydantic-settings python-dotenv

run:
  DETERMINISTIC=1 RNG_SEED=42 ENGINE_BACKEND=cpu_numpy \
  SNAPSHOT_BACKEND=disabled RAY_ENABLED=0 KAFKA_BROKERS="" \
  python -m src.main

verify-neuromorphic:
  # 2 незалежних прогони → порівняти StepOutputs → MSE < 1e-6
  DETERMINISTIC=1 RNG_SEED=42 ENGINE_BACKEND=cpu_numpy SNAPSHOT_BACKEND=disabled \
  pytest tests/test_numerical_safety.py tests/test_determinism.py -q
  # Якщо nan_inf_detected=True хоча б раз → EXIT 1

verify-sero:
  python sero_m0.py | grep -c "PASS" | xargs -I{} test {} -eq 7
  # Якщо вивід ≠ "7/7 PASS" → EXIT 1 з повідомленням яка перевірка впала

smoke: verify-neuromorphic verify-sero
  @echo "SMOKE: OK" || (echo "SMOKE: FAILED — перевір вивід вище" && exit 1)

clean:
  rm -rf __pycache__ .pytest_cache dist/ *.pyc

═══════════════════════════════════════════════════════════
ВИВІД — ТІЛЬКИ ЦЕ
═══════════════════════════════════════════════════════════

[повний Makefile — реальний, виконуваний, без коментарів якщо ≤20 рядків]

Production:        make run
Детермінізм+safety: make smoke
Цикл з нуля:       make install smoke → __ секунд

═══════════════════════════════════════════════════════════
ЦИКЛІЧНИЙ ЗВОРОТНИЙ ЗВ'ЯЗОК (якщо smoke fails)
═══════════════════════════════════════════════════════════

  nan_inf_detected=True        → Промпт 3: не спростив kernel достатньо
  MSE > 1e-6 між двома run     → Промпт 1: є недетермінований компонент (RNG не ізольований)
  T виходить за [T_min, T_0]  → Промпт 2: safety floor не є KEEP, а видалений помилково
  smoke > 30 сек               → Промпт 4: dev loop не оптимізований
  ImportError: kafka/ray/boto3 → Промпт 1: залежність вижила без доказу
```

---

## КАРТА ПОТОКУ

```text
                    ┌─────────────────────────────────┐
                    │  ВХІД: повний код + specs + deps │
                    └──────────────┬──────────────────┘
                                   │
                              ПРОМПТ 1
                     "Яке поле StepOutputs ламається?"
                                   │
                    ┌──────────────▼──────────────────┐
                    │  ЗАЛИШИТИ (≤7) + ЗНИЩИТИ + МЕТА  │
                    └──────────────┬──────────────────┘
                                   │
                              ПРОМПТ 2
                    "Доведи що не можна видалити — або KILL"
                                   │
                         Z < 10%? ─┤─ YES → повернись в П1
                                   │ NO
                    ┌──────────────▼──────────────────┐
                    │   KEEP таблиця верифікована      │
                    └──────────────┬──────────────────┘
                                   │
                              ПРОМПТ 3
                       "Спрости до рівняння, не framework"
                                   │
                      > 60 рядків? ┤─ YES → повернись в П2
                                   │ NO
                    ┌──────────────▼──────────────────┐
                    │  Фінальне ядро + метрики         │
                    └──────────────┬──────────────────┘
                                   │
                              ПРОМПТ 4
                        "Dev loop < 10 с або повернись"
                                   │
                      > 10 сек?   ─┤─ YES → повернись в П3
                                   │ NO
                    ┌──────────────▼──────────────────┐
                    │  Три рівні циклу верифіковані    │
                    └──────────────┬──────────────────┘
                                   │
                              ПРОМПТ 5
                     "Один Makefile ≤20 рядків або повернись"
                                   │
                    ┌──────────────▼──────────────────┐
                    │       make smoke                 │
                    └──────────────┬──────────────────┘
                                   │
              nan_inf? ────────────┤──────────────── → П3
              MSE>1e-6? ───────────┤──────────────── → П1
              T∉bounds? ───────────┤──────────────── → П2
              slow? ───────────────┤──────────────── → П4
              ImportError? ────────┤──────────────── → П1
                                   │ ALL PASS
                    ┌──────────────▼──────────────────┐
                    │         DONE. VERIFIED.          │
                    └─────────────────────────────────┘
```

---

*Версія v3. Калібровано до: step_inplace() / sero_step() / StepOutputs / clamp_V_inplace / nan_inf_detected / test_determinism.py / test_numerical_safety.py / DETERMINISTIC=1 RNG_SEED=42 ENGINE_BACKEND=cpu_numpy.*
