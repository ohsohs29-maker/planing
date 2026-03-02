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
    "numpy",
    "pandas",
    "plotly",
    "scipy",
    "matplotlib",
    "networkx",
]
# ─────────────────────────────────────────────────────


def get_python_cmd():
    """현재 실행 중인 Python 경로 반환"""
    return sys.executable


def get_venv_python():
    """가상환경 내 Python 경로 (OS별)"""
    if platform.system() == "Windows":
        return VENV_DIR / "Scripts" / "python.exe"
    else:
        return VENV_DIR / "bin" / "python"


def get_venv_pip():
    """가상환경 내 pip 경로 (OS별)"""
    if platform.system() == "Windows":
        return VENV_DIR / "Scripts" / "pip.exe"
    else:
        return VENV_DIR / "bin" / "pip"


def create_venv():
    """가상환경 생성"""
    if VENV_DIR.exists():
        print(f"✅ 가상환경이 이미 존재합니다: {VENV_DIR}")
        return

    print(f"📦 가상환경 생성 중: {VENV_DIR}")
    subprocess.check_call([get_python_cmd(), "-m", "venv", str(VENV_DIR)])
    print(f"✅ 가상환경 생성 완료!")


def install_packages():
    """패키지 설치"""
    pip = str(get_venv_pip())

    # pip 업그레이드
    print("\n📦 pip 업그레이드 중...")
    subprocess.check_call([pip, "install", "--upgrade", "pip", "-q"])

    # 패키지 설치
    print(f"📦 패키지 설치 중: {', '.join(PACKAGES)}")
    subprocess.check_call([pip, "install"] + PACKAGES + ["-q"])
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


def verify():
    """설치 확인"""
    venv_python = str(get_venv_python())

    print("\n🔍 설치 확인 중...")
    result = subprocess.run(
        [venv_python, "-c", f"""
import sys
print(f"  Python: {{sys.version}}")
pkgs = {PACKAGES!r}
ok, fail = [], []
for p in pkgs:
    try:
        __import__(p)
        ok.append(p)
    except ImportError:
        fail.append(p)
print(f"  설치 완료: {{', '.join(ok)}}")
if fail:
    print(f"  ❌ 설치 실패: {{', '.join(fail)}}")
else:
    print("  ✅ 모든 패키지 정상!")
"""],
        capture_output=False,
    )


def print_instructions():
    """사용 안내 출력"""
    print("\n" + "=" * 56)
    print("  설정 완료! 다음 단계를 따라주세요.")
    print("=" * 56)
    print()
    print("  1. VSCode에서 .ipynb 파일을 엽니다")
    print("  2. 우측 상단 [커널 선택] 클릭")
    print(f'  3. "{KERNEL_DISPLAY}" 선택')
    print("  4. 첫 번째 셀 실행 (▶)")
    print()
    if platform.system() == "Windows":
        activate = f"  {VENV_DIR}\\Scripts\\activate"
    else:
        activate = f"  source {VENV_DIR}/bin/activate"
    print(f"  (터미널에서 직접 활성화: {activate})")
    print("=" * 56)


def main():
    os_name = platform.system()
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    print("=" * 56)
    print("  AI 기획 강의 - 실습 환경 자동 설정")
    print("=" * 56)
    print(f"  OS: {os_name} ({platform.machine()})")
    print(f"  Python: {py_ver} ({get_python_cmd()})")
    print(f"  가상환경: {VENV_DIR}")
    print("=" * 56)

    if sys.version_info < (3, 10):
        print("❌ Python 3.10 이상이 필요합니다.")
        print(f"   현재 버전: {py_ver}")
        sys.exit(1)

    create_venv()
    install_packages()
    register_kernel()
    verify()
    print_instructions()


if __name__ == "__main__":
    main()
