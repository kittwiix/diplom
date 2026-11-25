# Ректификация фотографий стеллажа по гомографии

Дипломная работа: автоматическое выпрямление фото полки (наклон + перспектива) так, чтобы
ценники читались внешней VLM-моделью. Математика — своя (без OpenCV), нейросетевая часть — ONNX,
целевая платформа — Android (Java).

- план работ: [docs/PLAN.md](docs/PLAN.md)
- теория: `docs/THEORY.md` (в работе), контракт ONNX: `docs/SPEC-model-io.md` (в работе)

## Что уже работает (стадия 1)

```
фото -> кромки полок (профили + RANSAC) -> точки схода -> H = A·H_∞ -> свой варп -> кадр + полосы полок
```

- детектор кромок полок и вертикалей (субпиксельные пики, RANSAC-связывание, отбраковка по покрытию);
- аффинная + метрическая ректификация по точкам схода; режимы `vp` / `directions` / `roll`;
- собственный ресемплер: обратное отображение, билинейно, префильтр против алиасинга, тайлы;
- обрезка чёрных углов (`crop=valid`), нарезка на полосы-ячейки для VLM;
- метрика качества без разметки: остаточный угол кромок (`quality.rms_deg`, `score`);
- CLI с JSON-метаданными (H, точки схода, углы), отчёт-картинка, 5 тестов;
- **1.0–1.3 с на кадр 12 Мп**, остаточный угол 0.03–0.37°.

## Быстрый старт

```bash
python -m pip install -r python/requirements.txt
cd python
python -m pytest tests -q          # 5 тестов
python tools/demo_stage1.py        # кадры + полосы + compare -> ../out/stage1
python tools/make_report.py        # отчёт-картинки (для показа) -> ../out/report
```

CLI:

```bash
cd python
python -m rectif.cli -i ../photo.jpeg -o ../out/photo_rect.jpg \
    --mode vp --crop valid --json ../out/photo_rect.json \
    --cells-dir ../out/cells --debug-dir ../out/debug
```

Ключи: `--mode vp|directions|roll`, `--crop none|valid|shelf|auto`, `--max-side`, `--cells-dir`, `--json`, `--debug-dir`.

## Структура

```
docs/       PLAN.md, experiments/m0 (анализ эталонных пар)
python/
  rectif/   imageops, lines, geometry, warp, pipeline, debug, io, cli
  tools/    demo_stage1.py, make_report.py, (далее) label_tool.html, make_synthetic.py, export_onnx.py
  tests/    test_geometry.py, test_pipeline.py
ml/         (далее) обучение модели
models/     (далее) onnx + manifest
java/, android/  (далее) порт и приложение
```
