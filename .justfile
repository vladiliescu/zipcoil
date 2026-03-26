default:
    @just --list

check:
    uvx ruff check . && uvx ty check .