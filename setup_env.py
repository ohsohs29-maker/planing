"""
========================================================
강의 실습 환경 자동 설정 스크립트
========================================================

사용법:
  python setup_env.py

기능:
  1. .venv 가상환경 생성 (이미 있으면 건너뜀)
  2. 필요한 패키지 일괄 설치
  3. Jupyter 커널 등록 (VSCode에서 선택 가능)

지원 OS: Windows, macOS, Linux
Python 요구 사항: 3.10 이상
========================================================
"""

import subprocess
import sys
import platform
from pathlib import Path

# ── 설정 ──────────────────────────────────────────────
VENV_DIR = Path(__file__).parent / ".venv"
KERNEL_NAME = "strategy-lecture"
KERNEL_DISPLAY = "AI 기획 강의 (Python 3)"

PACKAGES = [
    "ipykernel",
    "nbformat",
    "numpy",
    "pandas",
    "plotly",
    "scipy",
    "matplotlib",
    "networkx",
]
# ─────────────────────────────────────────────────────


def get_venv_python():
    """가상환경 내 Python 경로 (OS별)"""
    if platform.system() == "Windows":
        return VENV_DIR / "Scripts" / "python.exe"
    else:
        return VENV_DIR / "bin" / "python"


def create_venv():
    """가상환경 생성 (불완전하면 재생성)"""
    venv_python = str(get_venv_python())

    if VENV_DIR.exists():
        if Path(venv_python).exists():
            print(f"✅ 가상환경이 이미 존재합니다: {VENV_DIR}")
            return
        else:
            print(f"⚠️ 가상환경이 불완전합니다. 재생성합니다...")
            import shutil
            shutil.rmtree(VENV_DIR)

    print(f"📦 가상환경 생성 중: {VENV_DIR}")
    subprocess.check_call([sys.executable, "-m", "venv", str(VENV_DIR)])
    print("✅ 가상환경 생성 완료!")


def install_packages():
    """패키지 설치"""
    python = str(get_venv_python())

    print("\n📦 pip 업그레이드 중...")
    subprocess.check_call([python, "-m", "pip", "install", "--upgrade", "pip", "-q"])

    print(f"📦 패키지 설치 중: {', '.join(PACKAGES)}")
    subprocess.check_call([python, "-m", "pip", "install"] + PACKAGES + ["-q"])
    print("✅ 패키지 설치 완료!")


def register_kernel():
    """Jupyter 커널 등록"""
    venv_python = str(get_venv_python())

    print(f"\n📦 Jupyter 커널 등록 중: {KERNEL_DISPLAY}")
    subprocess.check_call([
        venv_python, "-m", "ipykernel", "install",
        "--user",
        "--name", KERNEL_NAME,
        "--display-name", KERNEL_DISPLAY,
    ])
    print("✅ 커널 등록 완료!")


def main():
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    print("=" * 56)
    print("  AI 기획 강의 - 실습 환경 자동 설정")
    print("=" * 56)
    print(f"  OS: {platform.system()} ({platform.machine()})")
    print(f"  Python: {py_ver} ({sys.executable})")
    print(f"  가상환경: {VENV_DIR}")
    print("=" * 56)

    if sys.version_info < (3, 10):
        print("❌ Python 3.10 이상이 필요합니다.")
        print(f"   현재 버전: {py_ver}")
        sys.exit(1)

    create_venv()
    install_packages()
    register_kernel()

    print("\n" + "=" * 56)
    print("  🎉 설정 완료!")
    print("=" * 56)
    print()
    print("  1. VSCode에서 파일 → 폴더 열기 → C:\\planing 선택")
    print("  2. ch01/ch01.ipynb 열기")
    print("  3. 우측 상단 [커널 선택] 클릭")
    print(f'  4. "{KERNEL_DISPLAY}" 선택')
    print("  5. 첫 번째 코드 셀 실행 (▶)")
    print("=" * 56)


if __name__ == "__main__":
    main()
