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

## Giao diện đồ họa (GUI)

Mở giao diện chơi trực tiếp (Người vs AI, AI vs AI, đổi thuật toán/tham số động và hoàn tác `Undo` trên cả bàn 3×3 lẫn Caro):

```bash
python -m src.ui.gui_core
```

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

## So sánh quy tắc chọn nước cuối R4 và sinh toàn bộ hình vẽ (P3.2b)

Chạy thực nghiệm R4 (so sánh `robust`, `max`, `max_robust` trên bàn 3×3 và Caro 9×9) và sinh toàn bộ biểu đồ vector PDF (`F1.pdf`, `F2.pdf`, `F3.pdf`) cùng bảng `T4`, `T5`:

~~~sh
python -m experiments.run configs/p32b_r4_3x3.yaml --workers 8
python -m experiments.run configs/p32b_r4_9x9.yaml --workers 8
python -m experiments.plots
~~~

## So sánh các chính sách rollout L0–L4 R5 (P3.3a)

Chạy thực nghiệm R5 (so sánh L0–L4 cùng số vòng lặp `iterations = 200`, cùng thời gian `time_s = 0.08` s/nước trên bàn Caro 9×9, đo tốc độ mô phỏng/giây `sims/s` và đánh giá trên thế bẫy E1) để sinh các bảng dữ liệu của Bảng `T6`:

~~~sh
python -m experiments.run configs/p33a_r5_equal_iter.yaml --workers 8
python -m experiments.run configs/p33a_r5_equal_time.yaml --workers 4
python -m experiments.p33a_r5
~~~

## Đóng góp

| Thành viên | Mã việc |
| --- | --- |
| TV1 | P6.1a, P1.1a, P1.1b, P1.1c, P2.1, P3.1a, P3.1b |
| TV2 | P1.2, P2.2a, P2.2b, P3.2a, P3.2b, P6.2 |
| TV3 | P1.3a, P1.3b, P2.3a, P2.3b, P3.3a, P5.3a |
