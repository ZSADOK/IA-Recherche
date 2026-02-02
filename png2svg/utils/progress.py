# progress.py
def progress_bar(elapsed, total, score):
    pct = min(1.0, elapsed / total) * 100
    print(
        f"\r[{pct:5.1f}%] MSE={score:10.2f}",
        end="",
        flush=True
    )
