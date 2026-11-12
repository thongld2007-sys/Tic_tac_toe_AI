# Quy ước làm việc

## Git

- Mỗi việc một nhánh, đặt tên `tvN/<mã việc>-<mô tả ngắn>`, ví dụ `tv2/P2.2a-mcts-core`.
- Tiêu đề PR có mã việc, gán đúng người review theo bảng phân công.
- Không push thẳng lên `main`. Chỉ merge khi CI pass và có 1 người duyệt.
- Người review phải chạy thử code, không chỉ đọc lướt.
- Đổi interface (`src/games/base.py`, `src/agents/base.py`) hoặc schema CSV sau mốc M1 phải có đủ 2 người còn lại duyệt.

## Code

- Python ≥ 3.10. Import theo đường dẫn tuyệt đối từ gốc repo: `from src.games.base import GameState`.
- Trước khi mở PR: `make lint` và `make test` phải pass.
- Mỗi file mã nguồn `src/<gói>/<tên>.py` có file test `tests/test_<tên>.py`. Các module được
  kiểm chứng bằng cách đối chiếu với nhau có thể dùng chung một file test, đặt tên theo cả hai
  (ví dụ `tests/test_minimax_alphabeta.py`).
- Mọi nguồn ngẫu nhiên nhận `seed` hoặc `rng` từ bên ngoài, không dùng trạng thái ngẫu nhiên toàn cục.
- Dòng tối đa 100 ký tự (ruff kiểm tra).

## Thực nghiệm

- Kết quả chỉ được sinh từ runner (`experiments/run.py`). Không sửa tay file CSV, không nhập tay số vào report.
- CSV kết quả commit vào `experiments/results/`.

## Ai sở hữu thư mục nào

| Thư mục / file | Chủ sở hữu |
| --- | --- |
| `src/games/base.py`, `tictactoe.py`, `src/agents/base.py`, `minimax.py`, `alphabeta.py` | TV1 |
| `experiments/run.py`, `verify_never_lose.py`, `configs/`, `src/ui/gui_panel.py`, CI | TV1 |
| `src/games/gomoku.py`, `src/agents/mcts.py`, `src/ui/gui_core.py`, `experiments/stats.py`, `plots.py` | TV2 |
| `src/eval/patterns.py`, `src/agents/alphabeta_gomoku.py`, `rollouts.py`, `report/` | TV3 |
| `src/games/xiangqi.py` | TV1 (luật) · TV2 (agent) · TV3 (GUI, thực nghiệm) |
