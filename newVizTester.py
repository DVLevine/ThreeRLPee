import numpy as np, threelp
res = threelp.build_canonical_stride(1.0, 0.1, 0.6)
x = res.x_ref
states, actions = [], []

for _ in range(6):              # 6 strides
    states.append(x)
    a = np.zeros(8, dtype=np.float64)   # residuals
    actions.append(a)            # one action per transition
    x = res.A @ x + res.B @ (res.u_ref + a) + res.b

threelp.visualize_canonical_rollout(
  states=states,
  actions=actions,             # enables dense sampling
  ref_stride=res,
  params=threelp.ThreeLPParams.Adult(),
  dense_substeps=150,
  show_reference=True,
  show_policy=True,
  loop=False,
  raise_on_skip=True,
)
