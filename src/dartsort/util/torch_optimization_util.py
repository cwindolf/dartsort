import torch


def batched_levenberg_marquardt(
    x0,
    vgradfunc,
    vhess,
    extra_args=(),
    lambd=100.0,
    nu=10.0,
    scale_problem="hessian",
    max_steps=250,
    convergence_g=1e-10,
    convergence_err=1e-10,
    min_scale=1e-2,
    tikhonov=0.0,
):
    """A batched torch implementation of Levenberg-Marquardt

    Solve a batch of minimization problems, starting from the batch
    of initial conditions x0 with shape (n, p). Here, n is the number
    of problems (batch size) and p is the problem size (number of features).

    vgradfunc and vhess should be functions such that
     - vgradfunc(x, *extra_args) returns a tuple of the (n, p) gradient at x and the
       (n,) objective at x
     - vhess(x, *extra_args) returns the (n, p, p) tensor of Hessians at x
    You can easily produce these using torch's vmap, as done elsewhere
    in dartsort. extra_args is a tuple of (n, ...) tensors which you can
    use to maintain other per-problem variables that are needed to evaluate
    your objective, gradient, or Hessian.

    Arguments
    ---------
    x : tensor, shape (n, p) as described above
        The initial condition. This will not be written to.
    vgradfunc, vhess : functions
        Batched (grad, objective) and Hessian functions as described above
    extra_args : tuple
        Extra arguments passed to vgradfunc and vhess as described above
    lambd : float, default 100.0
        Initial Levenberg-Marquardt damping factor
    nu : float, default 10.0
        Damping update multiplier for bad subproblems, or inverse multiplier
        for good problems (lambd <- lambd * nu or lambd / nu)
    scale_problem : one of ("hessian", "none")
        If scale_problem == "hessian", then the Hessian's diagonal is used to
        inform the damping.
    max_steps : int, default 250
        Hard stopping condition
    convergence_g, convergence_err : floats
        Stopping conditions
    min_scale : float, default 0.01
        Stability term, this is the minimum amount of damping. So it sets the
        minimum Tikhonov-style damping that is always used. Can be set to 0
        for nice problems.
    tikhonov : float, default 0.0
        Extra damping

    Returns
    -------
    x : (n, p) tensor
        Optimization result
    nsteps : (n,) LongTensor
        Number of steps taken for each problem
    """
    x = x0.clone()
    n, p = x.shape
    min_scale = torch.tensor(min_scale, dtype=x.dtype, device=x.device)

    # initial objective
    g, f = vgradfunc(x, *extra_args)

    # state variables
    # number of steps taken in each of the n problems
    nsteps = torch.zeros(n, dtype=torch.int, device=x.device)
    # current indices of active problems (all to start)
    # we don't run all problems at every step, only those that haven't
    # yet converged
    active = torch.arange(n, device=x.device)
    # the Levenberg-Marquardt damping factor, which changes dynamically
    lambd = torch.full_like(f, lambd)

    # these variable names
    xa = x
    eargsa = extra_args

    for it in range(max_steps):
        # get hessians for all active problems
        H = vhess(xa, *eargsa)

        # determine damping factor
        if scale_problem == "hessian":
            diag = H.diagonal(dim1=1, dim2=2).abs()
            H.diagonal(dim1=1, dim2=2).add_(
                lambd[:, None] * torch.maximum(diag, min_scale)
            )
        elif scale_problem == "none":
            H.diagonal(dim1=1, dim2=2).add_(lambd[:, None])
        else:
            assert False

        # extra damping
        if tikhonov > 0:
            H.diagonal(dim1=1, dim2=2).add_(tikhonov)

        # find the search direction `d`
        # use cholesky to find which problems that are pos def
        # cholesky_solve to get search directions for those,
        # and use gradient elsewhere
        L, info = torch.linalg.cholesky_ex(H)
        # initialize search direction with neg gradient
        d = -g
        ill_conditioned = info != 0
        nice = info == 0
        # cholesky solve nice problems
        if nice.any():
            d[nice] = torch.cholesky_solve(d[nice][..., None], L[nice])[..., 0]
        # problems with issues need to be damped
        if ill_conditioned.any():
            d[ill_conditioned] /= lambd[ill_conditioned, None]

        # check success, step, update lambd
        gnew, fnew = vgradfunc(xa + d, *eargsa)
        df = fnew - f
        shrink = df <= 0
        ill_conditioned = ill_conditioned | ~shrink

        # accept shrink steps
        if shrink.any():
            x[active[shrink]] += d[shrink]
            f[shrink] = fnew[shrink]
            g[shrink] = gnew[shrink]

        # check stopping conditions
        converged_g = g.square().sum(dim=1) < convergence_g
        converged_f = df.abs() < convergence_err
        converged = shrink & converged_f & converged_g

        # update damping factor based on ill conditioned-ness and
        # whether the objective decreased
        lambd = torch.where(ill_conditioned, lambd * nu, lambd / nu)

        # update active set
        remain = ~converged
        active = active[remain]
        if not active.numel():
            break
        xa = x[active]
        eargsa = tuple(ea[remain] for ea in eargsa)
        lambd = lambd[remain]
        f = f[remain]
        g = g[remain]
        nsteps[active] += 1

    return x, nsteps
