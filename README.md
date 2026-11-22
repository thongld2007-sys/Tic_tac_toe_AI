# Tic_tac_toe_AI

Bài tập lớn AI 2026–2027: Minimax, Alpha-Beta và MCTS-UCT trên Tic-Tac-Toe 3×3 và caro.

## Cài đặt và chạy test

Cần Python ≥ 3.10.

```bash
make install
make lint
make test
```

Trên Windows không có `make`, chạy lệnh tương đương:

```bash
python -m pip install -r requirements.txt
python -m ruff check .
python -m pytest
```

## Chạy thực nghiệm

Mỗi thực nghiệm là một file YAML trong `configs/`. Runner cho các agent đấu nhau và ghi mỗi ván
một dòng vào `experiments/results/<name>.csv`:

```bash
python -m experiments.run configs/smoke_random_3x3.yaml
```

Thêm `--workers N` để chọn số tiến trình và `--out file.csv` để đổi nơi ghi. Cách viết cấu hình
và ý nghĩa từng cột CSV nằm ở đầu file [experiments/run.py](experiments/run.py).

Kiểm chứng "MCTS không bao giờ thua" trên 3×3 (bảng T3 và ngưỡng I\*) có script riêng, ghi ba file
`verify_never_lose*.csv` vào `experiments/results/`:

```bash
python -m experiments.verify_never_lose
```

Mọi bảng của report (T1–T7, B2/B3, B4) được sinh thành file LaTeX trong `report/tables/` bằng
lệnh cuối dưới đây: T1 được đo lại, các bảng khác đọc CSV trong `experiments/results/`. Bảng T2
(ablation Alpha-Beta trên caro 15×15, 1 s/nước) cần chạy thực nghiệm riêng trước, mất khoảng 5 phút
với 15 tiến trình:

```bash
python -m experiments.run configs/p5_t2_ab_ablation.yaml --workers 15
python -m experiments.report_tables
```

## Giao diện đồ họa (GUI)

Mở giao diện chơi trực tiếp (Người vs AI, AI vs AI, đổi thuật toán/tham số động và hoàn tác `Undo` trên cả bàn 3×3 lẫn Caro):

```bash
python -m src.ui.gui_core
```

Mở giao diện kèm bảng phân tích MCTS (`make demo` chạy lệnh này). Tab "Bảng phân tích" hiện heatmap
số lượt thăm `N` trên bàn cờ và bảng 5 nước đứng đầu tại gốc với `N`, `W`, `W/N`, UCT. Nút
"Mở thế E1 và phân tích" nạp thế bẫy E1 trên bàn 3×3 và chạy MCTS ngay:

```bash
python -m src.ui.gui_panel
```

Thêm `--preset E1` để mở sẵn thế E1, `--position file.txt` để nạp thế cờ từ file, và
`--headless-demo` để in bảng 5 nước ra terminal mà không mở cửa sổ. File thế cờ là danh sách nước
(`moves: 0,1 0,0 2,0 1,1`, X đi trước, mỗi nước là `hàng,cột` tính từ 0) hoặc sơ đồ bàn cờ dùng
`X`, `O`, `.`; chi tiết ở đầu file [src/ui/gui_panel.py](src/ui/gui_panel.py).

## Cấu trúc thư mục

```
src/games/      Trò chơi: interface GameState, Tic-Tac-Toe, caro, cờ tướng
src/agents/     Agent: Minimax, Alpha-Beta, MCTS, rollout
src/eval/       Hàm lượng giá theo mẫu cho caro
src/ui/         GUI: phần lõi và bảng phân tích
experiments/    Runner, thống kê, script vẽ hình; kết quả CSV nằm trong results/
configs/        Cấu hình YAML của các thực nghiệm
report/         Report LaTeX (IEEEtran)
tests/          Unit test (pytest)
```

Quy ước git, code và thực nghiệm: xem [CONTRIBUTING.md](CONTRIBUTING.md).

## Khảo sát UCT (P3.1b)

Chạy từ gốc repo sau khi cài requirements.txt:

~~~sh
python -m experiments.run configs/p31b_3x3.yaml --workers 8
python -m experiments.run configs/p31b_9x9.yaml --workers 8
python -m experiments.p31b_figures
~~~

Hai cấu hình dùng MCTS L0 đấu Random. Mỗi điểm có 200 ván, đổi bên sau mỗi ván.
Bàn 3×3 bắt đầu trống. Bàn caro 9×9 dùng hai nước mở đầu dùng chung cho từng cặp ván.
Kết quả từng ván nằm trong experiments/results/p31b_*.csv, hình F1/F2 nằm trong report/figures.
Tỷ lệ thắng là số ván MCTS thắng chia tổng số ván, với khoảng Wilson 95%.

Hình ghi c theo thang thưởng [0, 1]. Runner ghi c thực sự truyền cho agent trong a_params.
Vì Q_code = 2 Q_đề − 1, chọn c_code = 2 c_đề khiến UCT_code = 2 UCT_đề − 1,
giữ cùng thứ tự ưu tiên giữa các nước đã thăm.

## So sánh quy tắc chọn nước cuối R4, biến thể/cải tiến UCT (B2/B3) và sinh hình vẽ (P3.2b, P4.2a)

Chạy thực nghiệm R4 (so sánh `robust`, `max`, `max_robust`), thực nghiệm Bonus B2/B3 (các biến thể UCT `standard`, `no_sqrt`, `no_log`, `sqrt_n` và cải tiến `ucb1_tuned`, `progressive_bias`) trên bàn 3×3 và Caro 9×9, rồi sinh toàn bộ biểu đồ vector PDF (`F1.pdf`..`F4.pdf`) cùng các bảng thống kê:

~~~sh
python -m experiments.run configs/p32b_r4_3x3.yaml --workers 8
python -m experiments.run configs/p32b_r4_9x9.yaml --workers 8
python -m experiments.run configs/p42a_b2_b3_3x3.yaml --workers 8
python -m experiments.run configs/p42a_b2_b3_9x9.yaml --workers 8
python -m experiments.plots
~~~

## Alpha-Beta vs MCTS cùng thời gian trên bàn Caro 15×15 (P3.3b)

Chạy thực nghiệm đối đầu trực tiếp giữa `AlphaBetaGomoku` và `MCTS` (`L2`, `L4`, dùng `c = 1.0` trên thang `[-1, 1]` tức `c_đề = 0.5` theo `F2`) ở cùng ngân sách thời gian `0.5`, `1.0`, `2.0` s/nước trên bàn 15×15 (`swap_first=true`, `opening_plies=2`) và tổng hợp Bảng `T7` tách theo bên đi trước / đi sau:

~~~sh
python -m experiments.run configs/p33b_ab_vs_mcts.yaml --workers 8
python -m experiments.p33b_ab_vs_mcts
~~~

## Cờ tướng (P4.1b, P4.2b)

`src/games/xiangqi.py` cài luật cờ tướng theo cùng interface `GameState` nên MCTS và runner dùng
được ngay (`game: {type: xiangqi, max_plies: 300}`). Một nước đi là cặp `(ô đi, ô đến)` với ô
đánh số `hàng * 9 + cột`; `parse_iccs("h2e2")` và `move_to_iccs` đổi sang ký hiệu ICCS. Bên hết
nước hợp lệ thì thua; hòa khi 120 nước liền không ăn quân, lặp thế ba lần hoặc chạm `max_plies`.

Kiểm tra bộ sinh nước bằng perft (số nút lá của cây nước hợp lệ). Thế ban đầu cho 44, 1 920,
79 666 và 3 290 240 nút ở độ sâu 1–4, khớp số tham chiếu:

```bash
python -m src.games.xiangqi --perft 4
```

Thêm `--fen "<FEN>"` để đếm từ thế khác và `--divide` để tách số nút theo từng nước đầu.

Hai agent cờ tướng (`src/agents/xiangqi_agents.py`, P4.2b) gồm `AlphaBetaXiangqi` (lượng giá vật
chất + bảng vị trí PST, MVV-LVA, Quiescence Search, Iterative Deepening và Zobrist TT) và
`XiangqiMCTSAgent` (MCTS-UCT với Rollout L4 `xiangqi_l4` cắt sớm bằng `tanh(eval / tau)`). Kiểm
chứng tỷ lệ thắng của cả hai agent trước `RandomAgent`:

```bash
python -m experiments.run configs/p42b_xiangqi_vs_random.yaml --workers 8
```

Giao diện đồ họa Cờ tướng (`src/ui/xiangqi_gui.py`, P4.3b) hỗ trợ chơi Người vs AI, AI vs AI,
nạp thế cờ mẫu / FEN, hoàn tác (Undo) và xem lịch sử ICCS; đồng thời chạy bộ thực nghiệm so sánh
`AlphaBetaXiangqi` và `XiangqiMCTSAgent` (cùng thời gian 0,10 s / 0,20 s, mở rộng độ sâu và số
lượt mô phỏng, tách theo bên Đỏ đi trước / Đen đi sau):

```bash
python -m src.ui.xiangqi_gui
python -m experiments.run configs/p43b_xiangqi_exp.yaml --workers 6 --out experiments/results/p43b_xiangqi_exp.csv
python -m experiments.p43b_xiangqi
```

Kết quả ván đấu nằm tại `experiments/results/p43b_xiangqi_exp.csv` (360 ván) và bảng tổng hợp
thống kê (Wilson 95%, bootstrap 95%, kiểm định $z$ theo phương sai $H_0$, tách Đỏ/Đen) nằm tại
`experiments/results/p43b_xiangqi_summary.csv`. Lưu ý khi phân tích mức 0,10 s/nước: MCTS-L4 chỉ
kịp duyệt ~101 nút/nước (so với ~590 nút/nước của Alpha-Beta) và dùng quá ngân sách thời gian
khoảng 7% (~107,3%) do kiểm tra đồng hồ sau khi hoàn tất mỗi rollout L4.

## Đóng góp

| Thành viên | Mã việc |
| --- | --- |
| TV1 | P6.1a, P1.1a, P1.1b, P1.1c, P2.1, P3.1a, P3.1b, P6.1b, P4.1a, P4.1b, P5.1a, P5.1b |
| TV2 | P1.2, P2.2a, P2.2b, P3.2a, P3.2b, P4.2a, P4.2b, P6.2 |
| TV3 | P1.3a, P1.3b, P2.3a, P2.3b, P3.3a, P3.3b, P4.3a, P4.3b, P5.3a |
