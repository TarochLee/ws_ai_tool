import subprocess
import shlex
import sys
from pathlib import Path

MODEL = Path("../cache/models/ggml-base-q5_1.bin")
AUDIO = Path("../cache/config/samples_jfk.wav")

# 你本机 whisper.cpp 的可执行文件路径：
# 1) 如果你把 whisper.cpp 编译输出放在项目里，常见是：./build/bin/whisper-cli
# 2) 或者你自己把它放到了 PATH 里，就写 "whisper-cli"
WHISPER_CLI = Path("./build/bin/whisper-cli")

def main():
    if not MODEL.exists():
        print(f"[ERROR] Model not found: {MODEL}", file=sys.stderr)
        sys.exit(1)

    if not AUDIO.exists():
        print(f"[ERROR] Audio not found: {AUDIO}", file=sys.stderr)
        sys.exit(1)

    # 如果本地没有 ./build/bin/whisper-cli，就尝试用 PATH 里的 whisper-cli
    cli = str(WHISPER_CLI) if WHISPER_CLI.exists() else "whisper-cli"

    # jfk 是英文样例，语言用 en；如果你换成中文音频，把 -l 改成 zh
    cmd = [
        cli,
        "-m", str(MODEL),
        "-f", str(AUDIO),
        "-l", "en",
        "--print-colors", "0",
        "--no-timestamps", "0",
    ]

    print("[CMD]", " ".join(shlex.quote(x) for x in cmd))
    print("---------- ASR OUTPUT ----------")

    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    print(p.stdout)

    if p.returncode != 0:
        print(f"[ERROR] whisper-cli exited with code {p.returncode}", file=sys.stderr)
        sys.exit(p.returncode)

if __name__ == "__main__":
    main()