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
| TV1 | P6.1a |
| TV2 |  |
| TV3 |  |
