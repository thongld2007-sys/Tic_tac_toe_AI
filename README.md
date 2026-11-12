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

## Đóng góp

| Thành viên | Mã việc |
| --- | --- |
| TV1 | P6.1a, P1.1a, P1.1b, P1.1c, P3.1a, P2.1 |
| TV2 | P1.2, P2.2a |
| TV3 | P1.3a, P1.3b, P5.3a |
