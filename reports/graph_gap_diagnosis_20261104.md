# 图谱缺口诊断

只核对本地来源和派生图，不创建图边，不验证临床结论或云端回答质量。

| 问题 | 分类 | 路径数 | 下一步 |
| --- | --- | ---: | --- |
| 氯雷他定分散片我已经吃了，鼻塞还是好不了，医生开的布地奈德鼻喷雾剂可以两个一起用吗？还是说必须停掉一个？ | no_pair_link_in_checked_sources | 0 | 当前检查范围未找到两药连接依据；先核对或补充可靠来源，不能虚构相互作用边 |
| 阿司匹林肠溶片与布洛芬片的成分和相互作用 | graph_paths_available | 4 | 图路径已经返回；若最终仍拒答，检查 evidence_flow 和审核原因 |
| 我喷了布地奈德鼻喷雾剂好几天了，鼻子还是堵，而且今天鼻子里好像有点疼，这是药不管用还是我有其他问题 | graph_paths_available | 6 | 图路径已经返回；若最终仍拒答，检查 evidence_flow 和审核原因 |
| 患者有严重的青光眼，同时正在服用西替利嗪（或弗雷他定），因为鼻炎鼻塞严重，能不能加上布地奈德喷雾剂？ | entity_resolution_or_clarification | 0 | 先核对药名和患者实际制剂，避免按错误实体补边 |

## 氯雷他定分散片我已经吃了，鼻塞还是好不了，医生开的布地奈德鼻喷雾剂可以两个一起用吗？还是说必须停掉一个？

当前检查范围未找到两药连接依据；先核对或补充可靠来源，不能虚构相互作用边

共同提及的第三方对象：酮康唑。不能作为这两药相互作用的连接依据。

- 氯雷他定分散片：contraindications：source_missing_in_local_snapshot；dosage：text_only_by_design

- 布地奈德鼻喷雾剂：composition：source_missing_in_local_snapshot；contraindications：source_missing_in_local_snapshot；dosage：text_only_by_design

## 阿司匹林肠溶片与布洛芬片的成分和相互作用

图路径已经返回；若最终仍拒答，检查 evidence_flow 和审核原因

共同提及的第三方对象：呋塞米、地高辛、甲氨蝶呤。不能作为这两药相互作用的连接依据。

- 阿司匹林肠溶片：adverse_reactions：source_missing_in_local_snapshot；dosage：text_only_by_design

- 布洛芬片：dosage：text_only_by_design

## 我喷了布地奈德鼻喷雾剂好几天了，鼻子还是堵，而且今天鼻子里好像有点疼，这是药不管用还是我有其他问题

图路径已经返回；若最终仍拒答，检查 evidence_flow 和审核原因

- 布地奈德鼻喷雾剂：composition：source_missing_in_local_snapshot；contraindications：source_missing_in_local_snapshot；dosage：text_only_by_design

## 患者有严重的青光眼，同时正在服用西替利嗪（或弗雷他定），因为鼻炎鼻塞严重，能不能加上布地奈德喷雾剂？

先核对药名和患者实际制剂，避免按错误实体补边

- 盐酸西替利嗪分散片：composition：source_missing_in_local_snapshot；dosage：text_only_by_design

- 盐酸西替利嗪口服溶液：dosage：text_only_by_design

- 盐酸西替利嗪滴剂：dosage：text_only_by_design

- 盐酸西替利嗪片：dosage：text_only_by_design

- 盐酸西替利嗪糖浆：dosage：text_only_by_design

- 盐酸西替利嗪胶囊：drug_interactions：source_missing_in_local_snapshot；dosage：text_only_by_design

- 布地奈德鼻喷雾剂：composition：source_missing_in_local_snapshot；contraindications：source_missing_in_local_snapshot；dosage：text_only_by_design

JSON 报告保留来源网址、原文片段、行哈希与位置。未命中不代表安全；分类仅为工程排查候选。
