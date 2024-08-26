.PHONY:	develop
develop:
	poetry run maturin develop -m packages/circuit/Cargo.toml
	poetry run maturin develop -m packages/qulacs/Cargo.toml
