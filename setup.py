from pathlib import Path

from setuptools import find_packages, setup


BASE_DIR = Path(__file__).resolve().parent


def read_requirements(path: Path) -> list[str]:
    requirements = []
    if not path.exists():
        return requirements

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        requirements.append(line)

    return requirements


setup(
    name="tous-contract-analysis",
    version="0.1.0",
    description="Utilities and notebooks for Terms of Use contract analysis.",
    packages=find_packages(include=["tools", "tools.*"]),
    include_package_data=True,
    install_requires=read_requirements(BASE_DIR / "requirements.txt"),
    python_requires=">=3.10",
)
