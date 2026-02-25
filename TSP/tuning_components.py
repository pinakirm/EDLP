import torch
import numpy as np



def eval_bdmala(model, bdmala, x_init, test_steps, use_dula, return_dict_key = None):
    hops = []
    x_cur = x_init
    bdmala.a_s = []
    for _ in range(test_steps):
        x_new = bdmala.step(x_cur, model, use_dula=use_dula)
        if return_dict_key is not None:
            cur_hops = ((x_new[return_dict_key] != x_cur[return_dict_key]) * 1.0).sum(-1).mean().item()
        else:
            cur_hops = ((x_new != x_cur) * 1.0).sum(-1).mean().item()
        hops.append(cur_hops)
        x_cur = x_new
    a_s = np.mean(bdmala.a_s)
    hops_mean = np.mean(hops)
    return a_s, hops_mean, x_cur


def update_hyperparam_metrics(best_idx, *lists):
    to_ret = []
    for l in lists:
        to_ret.append(l[best_idx])
    return to_ret


# helper function for adaptive functions
# essentially, tests all the hyper-parameters of interest
def run_hyperparameters(
    model, bdmala, x_init, test_steps, param_update_func, param_list, use_dula, return_dict_key = None
):
    a_s = []
    hops = []
    x_potential = []
    for p in param_list:
        param_update_func(bdmala, p)
        cur_a_s, cur_hops, x_cur = eval_bdmala(
            model, bdmala, x_init, test_steps, use_dula, return_dict_key = return_dict_key
        )
        a_s.append(cur_a_s)
        hops.append(cur_hops)
        x_potential.append(x_cur)
    return a_s, hops, x_potential


def estimate_opt_pair_greedy(
    model,
    bdmala,
    x_init,
    range_max,
    range_min,
    budget=250,
    zoom_resolution=5,
    test_steps=10,
    init_bal=0.95,
    a_s_cut=0.5,
):
    itr = 0

    def step_update(sampler, alpha):
        sampler.step_size = alpha

    hist_a_s = []
    hist_alpha_max = []
    hist_hops = []
    x_cur = x_init
    abs_max = range_max
    abs_min = range_min
    bal_increment = (0.5 / (budget / (zoom_resolution))) / 2
    while itr < budget:
        steps_to_test = np.linspace(range_min, range_max, zoom_resolution)
        bdmala.bal = init_bal
        a_s_l, hops_l, x_potential_l = run_hyperparameters(
            model, bdmala, x_cur, test_steps, step_update, steps_to_test
        )
        eval_a_s_l = [np.abs(a - a_s_cut) for a in a_s_l]
        cur_step, a_s, hops, x_cur = update_hyperparam_metrics(
            np.argmin(eval_a_s_l), steps_to_test, a_s_l, hops_l, x_potential_l
        )
        itr += len(steps_to_test) * test_steps
        hist_a_s.append(a_s)
        hist_alpha_max.append(cur_step)
        hist_hops.append(hops)

        # updating range min and range max
        # case where we aren't getting close to target a_s with cur_bal

        if np.abs(a_s - a_s_cut) >= 0.1:
            init_bal = init_bal - bal_increment
            range_min = abs_min
            range_max = cur_step
        else:
            step_interval = np.abs(steps_to_test[0] - steps_to_test[1])
            range_min = max(cur_step - step_interval, 0)
            range_max = min(cur_step + step_interval, abs_max)
    cur_step = cur_step / 2
    hist_metrics = {"a_s": hist_a_s, "alpha_max": hist_alpha_max, "hops": hist_hops}
    return x_cur, cur_step, hist_metrics, itr


# new function for finding alpha max based on the fact that it is difficult to tell
# a priori what the target acceptance rate should be for alpha max
def estimate_opt_step_greedy(
    model,
    bdmala,
    x_init,
    range_max,
    range_min,
    budget=250,
    zoom_resolution=5,
    test_steps=10,
    init_bal=0.95,
    a_s_cut=None,
):
    itr = 0

    def step_update(sampler, alpha):
        sampler.step_size = alpha

    hist_a_s = []
    hist_alpha_max = []
    hist_hops = []

    x_cur = x_init
    while itr < budget:
        steps_to_test = np.linspace(range_min, range_max, zoom_resolution)
        a_s_l, hops_l, x_potential_l = run_hyperparameters(
            model, bdmala, x_cur, test_steps, step_update, steps_to_test
        )
        if a_s_cut is not None:
            eval_a_s_l = [np.abs(a - a_s_cut) for a in a_s_l]
            cur_step, a_s, hops, x_cur = update_hyperparam_metrics(
                np.argmin(eval_a_s_l), steps_to_test, a_s_l, hops_l, x_potential_l
            )
        else:
            cur_step, a_s, hops, x_cur = update_hyperparam_metrics(
                np.argmax(a_s_l), steps_to_test, a_s_l, hops_l, x_potential_l
            )
        itr += len(steps_to_test) * test_steps
        hist_a_s.append(a_s)
        hist_alpha_max.append(cur_step)
        hist_hops.append(hops)

        # updating range min and range max
        step_interval = np.abs(steps_to_test[0] - steps_to_test[1])
        range_min = max(cur_step - step_interval, range_min)
        range_max = cur_step + step_interval
    cur_step = cur_step / 2
    hist_metrics = {"a_s": hist_a_s, "alpha_max": hist_alpha_max, "hops": hist_hops}
    return x_cur, cur_step, hist_metrics, itr


def estimate_alpha_max(
    model,
    bdmala,
    x_init,
    budget,
    init_step_size,
    a_s_cut=0.6,
    lr=0.5,
    test_steps=10,
    init_bal=0.95,
    est_resolution=1,
    error_margin=0.01,
    use_dula=False,
    step_update=None,
    return_dict_key = None
):
    a_s = 0
    cur_step = init_step_size
    hist_a_s = []
    hist_alpha_max = []
    hist_hops = []

    x_cur = x_init
    itr = 0
    bdmala.bal = init_bal
    if step_update is None:

        def step_update(sampler, alpha):
            sampler.step_size = alpha

    while itr < budget:
        proposal_step = cur_step * (1 - lr * np.abs(a_s - a_s_cut))
        steps_to_test = [proposal_step, cur_step]
        # steps_to_test = np.linspace(proposal_step, cur_step, est_resolution)
        a_s_l, hops_l, x_potential_l = run_hyperparameters(
            model,
            bdmala,
            x_cur,
            test_steps,
            step_update,
            steps_to_test,
            use_dula=use_dula,
            return_dict_key=return_dict_key
        )
        cur_step, a_s, hops, x_cur = update_hyperparam_metrics(
            np.argmax(a_s_l), steps_to_test, a_s_l, hops_l, x_potential_l
        )
        itr += test_steps * len(steps_to_test)
        hist_a_s.append(a_s)
        hist_alpha_max.append(cur_step)
        hist_hops.append(hops)
        if a_s > a_s_cut - error_margin:
            break
        # modified to see if this improves performance and avoids the case where it just continually decreases and gets smaller, worsening acceptance rate
    eval_a_s_l = [a_s - a_s_cut for a_s in hist_a_s]
    step_idx = np.argmax(eval_a_s_l)
    final_step_size = hist_alpha_max[step_idx]
    # final_step_size = cur_step
    hist_metrics = {
        "a_s": hist_a_s,
        "hops": hist_hops,
        "alpha_max": hist_alpha_max,
        "best_idx": step_idx,
    }
    return x_cur, final_step_size, hist_metrics, itr


def estimate_alpha_min(
    model,
    bdmala,
    x_cur,
    budget,
    init_step_size,
    test_steps=10,
    lr=0.5,
    a_s_cut=0.5,
    error_margin=0.01,
    init_bal=0.5,
    est_resolution=4,
    use_dula=False,
    step_update=None,
    return_dict_key=None
):
    a_s = 0
    cur_step = init_step_size
    # book keeping lists
    hist_a_s = []
    hist_alpha_min = []
    hist_hops = []
    # initialization for best acceptance rate, hops
    itr = 0
    bdmala.bal = init_bal
    if step_update is None:

        def step_update(sampler, alpha):
            sampler.step_size = alpha

    while itr < budget:
        proposal_step = cur_step * (1 + lr * np.abs(a_s - a_s_cut))
        # steps_to_test = [proposal_step, cur_step]
        steps_to_test = np.linspace(proposal_step, cur_step, est_resolution)
        a_s_l, hops_l, x_potential_l = run_hyperparameters(
            model,
            bdmala,
            x_cur,
            test_steps,
            step_update,
            steps_to_test,
            use_dula=use_dula,
            return_dict_key=return_dict_key
        )
        eval_a_s_l = [np.abs(a - a_s_cut) for a in a_s_l]
        # we want the smallest_step size possible
        cur_step, a_s, hops, x_cur = update_hyperparam_metrics(
            np.argmin(eval_a_s_l), steps_to_test, a_s_l, hops_l, x_potential_l
        )
        hist_a_s.append(a_s)
        hist_alpha_min.append(cur_step)
        hist_hops.append(hops)
        itr += len(steps_to_test) * test_steps

    eval_a_s_l = [np.abs(a_s - a_s_cut) for a_s in hist_a_s]
    step_idx = np.argmin(eval_a_s_l)
    final_step_size = hist_alpha_min[step_idx]
    final_step_size = cur_step
    hist_metrics = {
        "a_s": hist_a_s,
        "hops": hist_hops,
        "alpha_min": hist_alpha_min,
        "best_idx": step_idx,
    }
    return x_cur, final_step_size, hist_metrics, itr


# TODO: try doing cyclical sched on the balancing constant, not the step size
def estimate_opt_sched(
    model,
    bdmala,
    x_cur,
    opt_bal,
    alpha_max,
    a_s_cut,
    alpha_min,
    test_steps=10,
    est_resolution=10,
):
    hist_a_s = []
    hist_hops = []

    def update_step(sampler, alpha):
        sampler.step_size = alpha

    prev_step = alpha_max
    opt_steps = []
    for i in range(1, len(opt_bal) - 1):
        bal_val = opt_bal[i]
        bdmala.bal = bal_val
        proposal_steps = np.linspace(alpha_min, prev_step, est_resolution)
        a_s_l, hops_l, x_potential_l = run_hyperparameters(
            model,
            bdmala,
            x_cur,
            test_steps,
            update_step,
            proposal_steps,
            use_dula=False,
        )
        eval_a_s_l = [np.abs(a - a_s_cut) for a in a_s_l]
        cur_step, a_s, hops, x_cur = update_hyperparam_metrics(
            np.argmin(eval_a_s_l), proposal_steps, a_s_l, hops_l, x_potential_l
        )
        hist_a_s.append(a_s)
        hist_hops.append(hops)
        opt_steps.append(cur_step)
        prev_step = cur_step
    opt_steps = [alpha_max] + opt_steps + [alpha_min]
    metrics = {"a_s": hist_a_s, "hops": hist_hops}
    return x_cur, opt_steps, metrics


def estimate_opt_bal(
    model,
    bdmala,
    x_init,
    opt_steps,
    test_steps=10,
    init_bal=0.95,
    est_resolution=3,
    use_dula=False,
    init_small_bal=.5,
    return_dict_key=None,
):
    # we will calculate in REVERSE order
    if type(opt_steps) == list:
        alpha_to_use = opt_steps[::-1]
    elif type(opt_steps) == torch.Tensor:
        alpha_to_use = opt_steps.flip(dims=(0,))
    opt_bal = []
    hist_a_s = []
    hist_hops = []
    x_cur = x_init
    bal_proposals = [init_small_bal, init_small_bal + .05, init_small_bal + .1]

    def update_bal(sampler, beta):
        sampler.bal = beta

    for i, alpha in enumerate(alpha_to_use):
        bdmala.step_size = alpha
        a_s_l, hops_l, x_potential_l = run_hyperparameters(
            model,
            bdmala,
            x_cur,
            test_steps,
            update_bal,
            bal_proposals,
            use_dula=use_dula,
            return_dict_key=return_dict_key
        )
        best_bal, a_s, hops, x_cur = update_hyperparam_metrics(
            np.argmax(a_s_l), bal_proposals, a_s_l, hops_l, x_potential_l
        )
        hist_a_s.append(a_s)
        hist_hops.append(hops)
        opt_bal.append(best_bal)
        if best_bal >= 0.9:
            num_to_interpolate = len(opt_steps) - len(opt_bal)
            opt_bal = opt_bal + list(
                np.linspace(best_bal, init_bal, num_to_interpolate)
            )
            break
        else:
            bal_proposals = np.linspace(
                best_bal, min(init_bal, best_bal + 0.25), est_resolution
            )
    # interpolate the rest
    opt_bal = opt_bal[::-1]
    hist_metrics = {"a_s": hist_a_s, "hops": hist_hops}
    return x_cur, opt_bal, hist_metrics


class BayesOptimizer:
    def __init__(
        self,
        model: torch.nn.Module,
        sampler,
        test_steps,
        target_a_s=0.5,
        alpha_max_suggest=30,
        alpha_min_suggest=0.2,
        bal_optimizers=10,
    ):
        self.model = model
        self.bdmala = sampler
        self.test_steps = test_steps
        self.x_pool = []
        self.a_s_log = []
        self.hops_log = []
        self.target_a_s = target_a_s
        self.min_optimizer = bayes_opt.BayesianOptimization(
            f=self.acceptance_function,
            pbounds={"alpha": (0, 1), "beta": (0.5, 0.5)},
            verbose=2,
            random_state=1,
            allow_duplicate_points=True,
        )
        self.max_optimizer = bayes_opt.BayesianOptimization(
            f=self.acceptance_function,
            pbounds={"alpha": (0.05, 30), "beta": (0.8, 0.95)},
            verbose=2,
            random_state=1,
            allow_duplicate_points=True,
        )
        self.min_optimizer.probe({"alpha": 0.2, "beta": 0.5})
        self.bal_opts = []
        for i in range(bal_optimizers):
            tmp_opt = bayes_opt.BayesianOptimization(
                f=self.acceptance_function,
                verbose=2,
                pbounds={"alpha": (0.01, 30), "beta": (0.5, 0.95)},
                random_state=1,
                allow_duplicate_points=True,
            )
            self.bal_opts.append(tmp_opt)

    def _refresh_logs(self):
        self.x_pool = []
        self.a_s_log = []
        self.hops_log = []

    def acceptance_function(self, alpha, beta):
        self.bdmala.bal = beta
        self.bdmala.step_size = alpha
        x_to_use = self.x_pool[-1]
        a_s, hops_mean, x_cur = eval_bdmala(
            self.model, self.bdmala, x_to_use, self.test_steps
        )
        self.a_s_log.append(a_s)
        self.x_pool.append(x_cur)
        self.hops_log.append(hops_mean)
        obj = -np.abs(a_s - self.target_a_s)
        # obj = -np.abs(a_s - 0.75)
        # obj = hops_mean
        return obj

    def find_alpha_min(
        self,
        x_init,
        budget=250,
        init_points=0,
        noise=0,
    ):
        self._refresh_logs()
        self.x_pool.append(x_init)
        # self.min_optimizer.probe(
        #     params={"alpha": alpha_suggest, "beta": 0.5}, lazy=True
        # )
        # self.min_optimizer.set_gp_params(alpha=noise)
        self.min_optimizer.maximize(
            init_points=init_points,
            n_iter=(budget // self.test_steps - init_points),
        )

        # getting the best value for the current bound, setting kappa to 0

        return (
            self.min_optimizer.max["params"]["alpha"],
            self.a_s_log,
            self.hops_log,
            self.x_pool[-1],
        )

    def find_max_pair(self, x_init, budget=250, init_points=0, noise=0):
        self._refresh_logs()
        self.x_pool.append(x_init)
        # self.max_optimizer.set_gp_params(alpha=noise)
        self.max_optimizer.maximize(
            init_points=init_points,
            n_iter=(budget // self.test_steps) - init_points,
        )
        return (
            self.max_optimizer.max["params"]["alpha"],
            self.max_optimizer.max["params"]["beta"],
            self.a_s_log,
            self.hops_log,
            self.x_pool[-1],
        )

    def find_step_schedule(
        self, x_init, bal_sched, alpha_max, alpha_min, budget=320, init_points=5
    ):
        opt_budget = budget // len(bal_sched)
        self._refresh_logs()
        self.x_pool.append(x_init)
        opt_step_sched = [alpha_max]
        prev_alpha = alpha_max
        for i, b_opt in enumerate(self.bal_opts):
            bal_val = bal_sched[i + 1].cpu().numpy()
            b_opt.set_bounds(
                {"alpha": (alpha_min, prev_alpha), "beta": (bal_val, bal_val)}
            )
            b_opt.maximize(init_points=init_points, n_iter=opt_budget)
            alpha_val = b_opt.max["params"]["alpha"]
            opt_step_sched.append(alpha_val)
            prev_alpha = alpha_val
        opt_step_sched.append(alpha_min)
        return opt_step_sched