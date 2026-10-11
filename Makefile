.PHONY: test check-env base prepare-verl grpo-smoke

test:
	PYTHONPATH=src PYTHONPYCACHEPREFIX=work/pycache python -m unittest discover -s tests -v
	bash -n scripts/*.sh

check-env:
	python scripts/check_environment.py --require-gpu

base:
	bash scripts/run_base.sh

prepare-verl:
	python scripts/prepare_verl_data.py

grpo-smoke:
	bash scripts/run_grpo_smoke.sh

