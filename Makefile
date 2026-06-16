.PHONY: ci ci-fast ci-list precheck

ci:
	python3 scripts/run_ci_local.py

ci-fast:
	python3 scripts/run_ci_local.py --fast

ci-list:
	python3 scripts/run_ci_local.py --list

precheck: ci-fast
