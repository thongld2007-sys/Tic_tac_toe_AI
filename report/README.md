# Báo cáo Bài tập lớn AI 2026–2027 (IEEEtran LaTeX)

Thư mục này chứa khung báo cáo chuẩn `IEEEtran` (nhiệm vụ **P5.3a** — chủ sở hữu: **TV3**, review: **TV1**).
Mỗi thành viên viết phần của mình trong các file `.tex` riêng biệt tại `sections/` để tránh xung đột khi merge.

## Phân công file theo thành viên

| File | Chủ sở hữu | Nhiệm vụ | Nội dung chính |
| --- | --- | --- | --- |
| `main.tex` | TV3 | P5.3a, P5.3b | Khung `IEEEtran`, tiêu đề, tác giả, ghép toàn bộ các mục |
| `sections/00_abstract.tex` | TV3 | P5.3b | Tóm tắt (Abstract) |
| `sections/01_introduction.tex` | TV3 | P5.3a, P5.3b | I. Giới thiệu và 6 câu hỏi nghiên cứu **RQ1–RQ6** |
| `sections/02_related_work.tex` | TV1 | P5.1a | II. Nghiên cứu liên quan |
| `sections/03_problem_formulation.tex` | TV1 | P5.1a, P4.1a | III. Phát biểu bài toán và phân tích toán học UCT (B1) |
| `sections/04a_alphabeta.tex` | TV3 | P5.3b | IV-A. Cắt tỉa Alpha-Beta và hàm lượng giá mẫu cho Caro |
| `sections/04b_mcts_uct.tex` | TV2 | P5.2a | IV-B. Lõi MCTS-UCT, cách chọn nước cuối và biến thể UCT (B2, B3) |
| `sections/04c_rollout_policies.tex` | TV3 | P5.3b | IV-C. Chính sách Rollout L0–L4 và cải tiến mô phỏng (B4) |
| `sections/05_experimental_setup.tex` | TV3 | P5.3b | V. Thiết lập thực nghiệm và giao thức thống kê |
| `sections/06a_results_tv1.tex` | TV1 | P5.1a | VI-A. Kết quả T1, T3, F1, F2 (RQ1, RQ3, RQ4) |
| `sections/06b_results_tv2.tex` | TV2 | P5.2a | VI-B. Kết quả T4, T5 và B2/B3 (RQ5) |
| `sections/06c_results_tv3.tex` | TV3 | P5.3b | VI-C. Kết quả T2, T6, T7 (RQ2, RQ6) |
| `sections/07a_discussion_tv2.tex` | TV2 | P5.2a | VII-A. Thảo luận: chọn nước theo `N` so với `W/N` |
| `sections/07b_discussion_tv3.tex` | TV3 | P5.3b | VII-B. Thảo luận: Policy và Value, liên hệ AlphaGo, hạn chế |
| `sections/08_conclusion.tex` | TV3 | P5.3b | VIII. Kết luận |
| `references.bib` | TV1 | P5.1b | Danh mục tài liệu tham khảo (BibTeX) |

## Biên dịch trên Overleaf hoặc máy cá nhân

1. **Trên Overleaf**: Tải toàn bộ thư mục `report/` lên một project Overleaf mới (hoặc đồng bộ từ GitHub), chọn trình biên dịch **pdfLaTeX** và file gốc là `main.tex`.
2. **Trên máy cá nhân (có TeX Live / MacTeX)**:
   ```bash
   cd report
   pdflatex main.tex
   bibtex main
   pdflatex main.tex
   pdflatex main.tex
   ```
