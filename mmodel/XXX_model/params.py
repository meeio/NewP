from ..basic_params import basic_parser as p

p.add_argument("--batch_size", type=int)

p.add_argument("--depth", type=int)

p.add_argument("--widen_factor", type=int)

p.add_argument("--drop_rate", type=float)

p.add_argument("--monte_carlo_simuls", type=int)

p.add_argument("--eval_sample_num", type=int)

params = p.parse_args()