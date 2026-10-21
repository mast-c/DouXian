# -*- coding: utf-8 -*-
"""本地预览 V2.9 快速查询规划，不调用模型、检索或图谱。"""
import argparse
from fast_component_planner_v29 import plan_two_layer_aspects

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question",required=True,help="待检验的用户原问题")
    args=parser.parse_args(argv)
    result=plan_two_layer_aspects(args.question)
    if not result:
        print("未命中新规则：将走 V2.8 原查询分解路径（如需分解，可能调用 Qwen）。")
        return 0
    print(f"已命中 V2.9 Python 规则：{len(result)} 个子问题（无需 Qwen）")
    for i,query in enumerate(result,1):
        print(f"  子问题 {i}: {query}")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
