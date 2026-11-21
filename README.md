# findex — пошуковий рушій на Python

Навчальний проєкт пошукового рушія, реалізований поетапно в межах 8 лабораторних робіт.

## Лабораторна робота 4: Типи, тести, пакування та CLI

[![CI](https://github.com/DmytroKoste/findex/actions/workflows/ci.yml/badge.svg)](https://github.com/DmytroKoste/findex/actions/workflows/ci.yml)

### Встановлення з wheel та запуск CLI

```bash
uv build
uv tool install dist/findex-0.4.0-py3-none-any.whl
findex --help

findex index data/wiki --out index.bin --positions
findex search index.bin "(київ OR львів) AND україна" --k 5
findex search index.bin '"посольство україни"' --json
findex stats index.bin
