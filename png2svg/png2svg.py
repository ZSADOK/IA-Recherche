# png2svg.py
from __future__ import annotations

import argparse

from io_utils.image import load_image_bgr
from io_utils.svg import export_svg
from utils.rng import seed_all

from core.engine_ga import GAEngine
from core.engine_greedy import GreedyEngine


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("png2svg")

    p.add_argument("--input", required=True, help="Input image path.")
    p.add_argument("--output", required=True, help="Output SVG path.")

    p.add_argument("--algo", choices=["ga", "greedy"], default="greedy",
                   help="Algorithm: 'ga' baseline, 'greedy' improved (recommended).")

    p.add_argument("--shape", default="mixed",
                   choices=["rectangle", "circle", "ellipse", "mixed"],
                   help="Allowed primitive type(s).")

    p.add_argument("--n", type=int, default=150, help="Number of shapes.")
    p.add_argument("--time", type=int, default=60, help="Time limit in seconds.")
    p.add_argument("--seed", type=int, default=None, help="Random seed.")

    p.add_argument("--no-viz", action="store_true", help="Disable OpenCV visualization.")
    p.add_argument("--scale", type=int, default=4,
                   help="Fitness scale factor (4 good quality, 6-8 faster).")

    p.add_argument("--candidates", type=int, default=45,
                   help="Greedy: candidates per added shape (25-70).")
    p.add_argument("--refine", type=float, default=0.60,
                   help="Greedy: fraction of time spent refining (0.4-0.8).")

    p.add_argument("--pop", type=int, default=20, help="GA: population size.")
    p.add_argument("--mut", type=float, default=0.25, help="GA: per-child mutation probability.")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    seed_all(args.seed)

    target = load_image_bgr(args.input)

    if args.algo == "greedy":
        engine = GreedyEngine(
            target_bgr=target,
            shape_mode=args.shape,
            n_shapes=int(args.n),
            time_limit=float(args.time),
            enable_viz=not args.no_viz,
            fitness_scale=int(args.scale),
            candidates_per_shape=int(args.candidates),
            refine_fraction=float(args.refine),
        )
    else:
        engine = GAEngine(
            target_bgr=target,
            shape_mode=args.shape,
            n_shapes=int(args.n),
            time_limit=float(args.time),
            enable_viz=not args.no_viz,
            fitness_scale=int(args.scale),
            population_size=int(args.pop),
            mutation_rate=float(args.mut),
        )

    best = engine.run()

    export_svg(
        path=args.output,
        genotype=best,
        width=engine.width,
        height=engine.height,
        background_bgr=engine.background_bgr,
    )

    print("\nFinished.")
    print(f"Algorithm: {args.algo}")
    print(f"Shapes: {len(best)}")
    print(f"SVG saved to: {args.output}")


if __name__ == "__main__":
    main()