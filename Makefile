PY ?= /usr/local/bin/python3.12
export PYTHONPATH := src

DATA ?= data
RESULTS ?= results
JSONL = $(RESULTS)/LARGE_SCALE_LOGIT_FIELD.jsonl
JSON = $(RESULTS)/LARGE_SCALE_LOGIT_FIELD.json
TEX = paper/supplement_large_scale_logit.tex

FROZEN_COMMIT = 7245cfb279b0aa9587c2a914e52b6cd077868d1c
MANIFEST_URL = https://raw.githubusercontent.com/shaneraphel/geoaudit-cryptobench/$(FROZEN_COMMIT)/data/cryptobench_apo/train_manifest.json
CACHE_URL = https://github.com/shaneraphel/geoaudit-cryptobench-scaleup/releases/download/v0.1-data/_wide_cache_train.npz

COMPAT_JSON = $(RESULTS)/COMPATIBLE_READOUT.json
COMPAT_TEX = paper/supplement_compatible.tex

.PHONY: selftest data digits reproduce-half run summarise check compatible clean

selftest:
	$(PY) -m pytest tests/ -q

data: $(DATA)/train_manifest.json $(DATA)/_wide_cache_train.npz

$(DATA)/train_manifest.json:
	mkdir -p $(DATA)
	curl -fsSL "$(MANIFEST_URL)" -o $@
	$(PY) -c "from scaleup.data import sha256_of_file; from scaleup.constants import MANIFEST_SHA256; d=sha256_of_file('$@'); assert d==MANIFEST_SHA256, d; print('manifest sha256 OK')"

$(DATA)/_wide_cache_train.npz:
	mkdir -p $(DATA)
	curl -fsSL "$(CACHE_URL)" -o $@
	$(PY) -c "from scaleup.data import sha256_of_file; from scaleup.constants import CACHE_SHA256; d=sha256_of_file('$@'); assert d==CACHE_SHA256, d; print('cache sha256 OK')"

digits: $(DATA)/_digits_train.npy

$(DATA)/_digits_train.npy: $(DATA)/_wide_cache_train.npz
	$(PY) -c "from scaleup.digits import digitise_to_file; from scaleup.constants import LEVELS; digitise_to_file('$(DATA)/_wide_cache_train.npz', '$@', LEVELS); print('digits OK')"

reproduce-half: digits $(DATA)/train_manifest.json
	$(PY) -m scaleup.run run-half --data-dir $(DATA) --out $(JSONL) --seed 20260725 --direction forward

run: reproduce-half
	$(PY) -m scaleup.run run-all --data-dir $(DATA) --out $(JSONL) --json $(JSON) --tex $(TEX)

summarise:
	$(PY) -m scaleup.run summarise --data-dir $(DATA) --jsonl $(JSONL) --json $(JSON) --tex $(TEX)

check:
	$(PY) -m scaleup.run check --jsonl $(JSONL) --json $(JSON)

compatible: digits $(DATA)/train_manifest.json
	$(PY) -m scaleup.compatible --data-dir $(DATA) --json $(COMPAT_JSON) --tex $(COMPAT_TEX)

clean:
	rm -f $(JSONL) $(JSON) $(TEX)
