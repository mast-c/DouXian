"""一键运行整合包内的离线测试，不调用真实 DashScope 服务。"""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
TESTS = ['test_kb_fast_strict_v24.py', 'test_fast_query_planner_v23.py', 'test_method_coverage_v2.py', 'test_rerank_token_opt_v1.py', 'test_guard_fast_toggle.py', 'test_evidence_guard_v11.py', 'test_evidence_guard.py', 'test_self_rag_v15.py']

def main():
    failures = []
    for name in TESTS:
        process = subprocess.run(
            [sys.executable, "-S", name],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        print(f"\n=== {name} ===")
        output = process.stdout.strip()
        if output:
            print(output[-450:])
        if process.returncode:
            failures.append(name)
            print(process.stderr[-1500:])
    if failures:
        print(f"\n离线回归失败：{len(failures)} 项 -> {failures}")
        return 1
    print(f"\n全部 {len(TESTS)} 份离线测试脚本已通过。")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
