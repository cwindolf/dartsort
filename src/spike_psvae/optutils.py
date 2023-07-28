import torch
from torch.optim.lbfgs import _strong_wolfe


def single_newton(
    x,
    grad_and_func,
    hess,
    extra_args=(),
    lr=1.0,
    nsteps=100,
    convergence_x=1e-10,
    tikhonov=0.0,
    max_ls=25,
    c1=1e-4,
):
    nevalstot = 1
    g, f = grad_and_func(x, *extra_args)

    def ls_closure(x, t, d):
        g_, f_ = grad_and_func(x + t * d)
        return f_, g_

    for i in range(nsteps):
        H = hess(x)
        if tikhonov > 0:
            H.diagonal().add_(tikhonov)
        L, info = torch.linalg.cholesky_ex(H)
        if info == 0:
            d = torch.cholesky_solve(-g[:, None], L)
            d = d[:, 0]
        else:
            # if npd, just use the gradient :P
            d = -g

        gtd = g.dot(d)

        fnew, gnew, t, nevals = _strong_wolfe(
            ls_closure, x, lr, d, f, g, gtd, max_ls=max_ls, c1=c1
        )

        nevalstot += nevals
        dx = t * d
        if torch.linalg.norm(dx) < convergence_x:
            break
        x = x + dx
        f = fnew
        g = gnew
    return x, nevalstot, i


def batched_newton(
    x,
    vgradfunc,
    vhess,
    extra_args=(),
    lr=1.0,
    max_steps=100,
    convergence_x=1e-10,
    convergence_g=1e-7,
    tikhonov=0.0,
    method="cholesky",
    max_ls=25,
    wolfe_c1=1e-4,
    wolfe_c2=0.9,
):
    x = x.clone()
    n, p = x.shape
    nfevals = torch.ones(n, dtype=torch.int, device=x.device)
    nsteps = torch.zeros(n, dtype=torch.int, device=x.device)
    active = torch.arange(n, device=x.device)

    g, f = vgradfunc(x, *extra_args)
    xa = x
    eargsa = extra_args
    lra = torch.full_like(f, lr)

    for it in range(max_steps):
        # get hessians for all active problems
        H = vhess(xa, *eargsa)

        if tikhonov > 0:
            H.diagonal(dim1=1, dim2=2).add_(tikhonov)

        # find the search direction `d`
        if method == "cholesky":
            # use cholesky to find which Hessians are pd
            # cholesky_solve to get search directions for those,
            # and use gradient elsewhere
            L, info = torch.linalg.cholesky_ex(H)
            d = -g
            nice_hessian = info == 0
            if nice_hessian.any():
                d[nice_hessian] = torch.cholesky_solve(
                    d[nice_hessian][..., None], L[nice_hessian]
                )[..., 0]
        elif method == "pinv":
            # not working as well as chol
            d = torch.einsum("nij,nj->ni", H.pinverse(), -g)
        else:
            assert False

        # line search along `d`
        gtd = (g * d).sum(1)
        fnew, gnew, t, nevals = batched_strong_wolfe(
            vgradfunc,
            xa,
            lra,
            d,
            f,
            g,
            gtd,
            extra_args=extra_args,
            max_ls=max_ls,
            c1=wolfe_c1,
            c2=wolfe_c2,
        )
        nfevals[active] += nevals
        dx = t[:, None] * d
        x[active] += dx
        nsteps[active] += 1

        # check stopping conditions
        converged_x = dx.square().sum(dim=1) < convergence_x
        converged_g = gnew.square().sum(dim=1) < convergence_g
        converged = converged_x & converged_g

        # update active set
        remain = ~converged
        active = active[remain]
        if not active.numel():
            break
        xa = x[active]
        eargsa = tuple(ea[remain] for ea in eargsa)
        lra = lra[remain]
        f = fnew[remain]
        g = gnew[remain]

    return x, nfevals, nsteps


def batched_levenberg_marquardt(
    x,
    vgradfunc,
    vhess,
    extra_args=(),
    lambd=100.,
    tikhonov=0.0,
    nu=10.0,
    max_steps=100,
    scale_problem="hessian",
    min_scale=1e-2,
    convergence_g=1e-7,
    convergence_err=1e-7,
):
    x = x.clone()
    n, p = x.shape
    nsteps = torch.zeros(n, dtype=torch.int, device=x.device)
    active = torch.arange(n, device=x.device)

    g, f = vgradfunc(x, *extra_args)
    xa = x
    eargsa = extra_args
    lambd = torch.full_like(f, lambd)
    min_scale = torch.tensor(min_scale, dtype=x.dtype, device=x.device)

    for it in range(max_steps):
        # get hessians for all active problems
        H = vhess(xa, *eargsa)

        if scale_problem == "hessian":
            diag = H.diagonal(dim1=1, dim2=2).abs()
            H.diagonal(dim1=1, dim2=2).add_(lambd[:, None] * torch.maximum(diag, min_scale))
        elif scale_problem == "none":
            H.diagonal(dim1=1, dim2=2).add_(lambd[:, None])
        else:
            assert False
        if tikhonov > 0:
            H.diagonal(dim1=1, dim2=2).add_(tikhonov)

        # find the search direction `d`
        # use cholesky to find which Hessians are pd
        # cholesky_solve to get search directions for those,
        # and use gradient elsewhere
        L, info = torch.linalg.cholesky_ex(H)
        d = -g
        ill = info != 0
        nice = info == 0
        if nice.any():
            d[nice] = torch.cholesky_solve(d[nice][..., None], L[nice])[..., 0]
        if ill.any():
            d[ill] /= lambd[ill, None]

        # check success, step, update lambd
        gnew, fnew = vgradfunc(xa + d, *eargsa)
        df = fnew - f
        shrink = df <= 0
        ill = ill | ~shrink

        # accept shrink steps
        if shrink.any():
            x[active[shrink]] += d[shrink]
            f[shrink] = fnew[shrink]
            g[shrink] = gnew[shrink]

        # check stopping conditions
        converged_g = g.square().sum(dim=1) < convergence_g
        converged_f = df.abs() < convergence_err
        converged = shrink & converged_f & converged_g

        # change lambd based on ill
        lambd = torch.where(ill, lambd * nu, lambd / nu)

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


# -- batched versions of torch lbfgs' line search routine


def batched_cubic_interpolate(x1, f1, g1, x2, f2, g2, bounds=None):
    """Argmin of cubic interpolation of a scalar function (batched)"""
    # bounds logic -- make sure x1 <= x2
    correct_order = x1 <= x2
    if bounds is None:
        xmin_bound = torch.where(correct_order, x1, x2)
        xmax_bound = torch.where(correct_order, x2, x1)
    else:
        xmin_bound, xmax_bound = bounds

    # Code for most common case: cubic interpolation of 2 points
    #   w/ function and derivative values for both
    # Solution in this case (where x2 is the farthest point):
    #   d1 = g1 + g2 - 3*(f1-f2)/(x1-x2);
    #   d2 = sqrt(d1^2 - g1*g2);
    #   min_pos = x2 - (x2 - x1)*((g2 + d2 - d1)/(g2 - g1 + 2*d2));
    #   t_new = min(max(min_pos,xmin_bound),xmax_bound);
    d1 = g1 + g2 - 3.0 * (f1 - f2) / (x1 - x2)
    d2_square = d1.square() - g1 * g2
    d2 = d2_square.abs().sqrt()
    min_pos = torch.where(
        correct_order,
        x2 - (x2 - x1) * ((g2 + d2 - d1) / (g2 - g1 + 2 * d2)),
        x1 - (x1 - x2) * ((g1 + d2 - d1) / (g1 - g2 + 2 * d2)),
    )
    min_pos = torch.where(
        d2_square >= 0,
        min_pos.clamp_(xmin_bound, xmax_bound),
        (xmin_bound + xmax_bound) / 2.0,
    )
    return min_pos


def batched_strong_wolfe(
    batched_grad_and_obj,
    x,
    t,
    d,
    f,
    g,
    gtd,
    extra_args=(),
    c1=1e-4,
    c2=0.9,
    tolerance_change=1e-9,
    max_ls=25,
):
    """Documented and batched version of torch's strong Wolfe line search routine

    torch's original lua is much better documented:
    https://github.com/torch/optim/blob/master/lswolfe.lua

    acknowledgements to the nice library and useful reference
    @rfeinman/pytorch-minimize/blob/master/torchmin/line_search.py

    Arguments
    ---------
    batched_grad_and_obj : function
    x : tensor
        Current center / starting point
    t : float
        Initial step size. In torch LBFGS they set this to `lr` (i.e. 1 for Newton methods)
        except on the first iterate, when they use `lr*min(1,l1norm(grad))`.
    d : tensor
        Descent direction
    f : initial objective function value
    g : gradient at x
    gtd : tensor
        Directional derivative at starting point (grad @ d)
    c1, c2, tolerance_change : floats
        parameters: sufficient decrease, curvature, minimum allowed step length
    max_ls : int
        Allowed number of iterations

    Returns
    -------
    f : function value at x+t*d
    g : gradient value at x+t*d
    t : step length
    ls_nevals : number of objective function evals
    """
    B, in_dim = x.shape
    t = t.clone()
    # ported from https://github.com/torch/optim/blob/master/lswolfe.lua
    # via https://github.com/pytorch/pytorch/blob/main/torch/optim/lbfgs.py

    # override obj with extra args masked for the current active set
    def active_grad_and_obj(x, active_set):
        return batched_grad_and_obj(x, *(ea[active_set] for ea in extra_args))

    # to make this work in batched setting, we do two structural changes:
    # - none of these length-1 brackets. bracket is always two numbers.
    # - "active set" idea. breaks after condition checks are implemented
    #   by deactivating some batch members.

    done = torch.zeros(B, dtype=torch.bool, device=t.device)
    active_backtrack = torch.arange(B, device=t.device)
    ls_iter = torch.zeros(B, dtype=torch.int, device=t.device)

    d_norm, _ = d.abs().max(dim=1)
    g = g.clone(memory_format=torch.contiguous_format)
    # evaluate objective and gradient using initial step
    g_new, f_new = active_grad_and_obj(x + t[:, None] * d, active_backtrack)
    g_new = g_new.clone(memory_format=torch.contiguous_format)
    ls_func_evals = torch.ones(B, dtype=torch.int, device=t.device)
    gtd_new = (g_new * d).sum(dim=1)

    # bracket an interval containing a point satisfying the Wolfe criteria
    t_prev = torch.zeros_like(t)
    f_prev = f
    g_prev = g
    gtd_prev = gtd

    # allocate brackets
    bracket_l = torch.empty(B, dtype=t.dtype, device=t.device)
    bracket_u = torch.empty(B, dtype=t.dtype, device=t.device)
    bracket_fl = torch.empty(B, dtype=t.dtype, device=t.device)
    bracket_fu = torch.empty(B, dtype=t.dtype, device=t.device)
    bracket_gl = torch.empty((B, in_dim), dtype=t.dtype, device=t.device)
    bracket_gu = torch.empty((B, in_dim), dtype=t.dtype, device=t.device)
    bracket_gtdl = torch.empty(B, dtype=t.dtype, device=t.device)
    bracket_gtdu = torch.empty(B, dtype=t.dtype, device=t.device)

    # cur active set
    fa = f
    gtda = gtd
    ta = t
    t_preva = t_prev
    f_preva = f_prev
    f_newa = f_new
    g_preva = g_prev
    g_newa = g_new
    gtd_preva = gtd_prev
    gtd_newa = gtd_new

    while (
        ls_iter[active_backtrack] < max_ls
    ).any() and active_backtrack.numel():
        # check conditions
        cond1 = (f_newa > (fa + c1 * ta * gtda)) | (
            (ls_iter[active_backtrack] > 1) & (f_newa >= f_preva)
        )
        cond2 = gtd_newa.abs() <= -c2 * gtda
        cond3 = gtd_newa >= 0
        conda = cond1 | cond2 | cond3

        # -- set brackets
        # conds 1 + 3
        conds = cond1 | cond3
        if conds.any():
            oinds = active_backtrack[conds]
            bracket_l[oinds] = t_preva[conds]
            bracket_u[oinds] = ta[conds]
            bracket_fl[oinds] = f_preva[conds]
            bracket_fu[oinds] = f_newa[conds]
            bracket_gl[oinds] = g_preva[conds]
            bracket_gu[oinds] = g_newa[conds]
            bracket_gtdl[oinds] = gtd_preva[conds]
            bracket_gtdu[oinds] = gtd_newa[conds]
        # cond 2
        if cond2.any():
            oinds = active_backtrack[cond2]
            done[oinds] = True
            bracket_l[oinds] = bracket_u[oinds] = ta[cond2]
            bracket_fl[oinds] = bracket_fu[oinds] = f_newa[cond2]
            bracket_gl[oinds] = bracket_gu[oinds] = g_newa[cond2]

        # update active set
        unconda = ~conda
        active_backtrack = active_backtrack[unconda]
        if not active_backtrack.numel():
            break
        fa = fa[unconda]
        gtda = gtda[unconda]
        f_newa = f_newa[unconda]
        g_newa = g_newa[unconda]
        gtd_newa = gtd_newa[unconda]

        # interpolate to find new argmin
        ta = ta[unconda]
        min_step = ta + 0.01 * (ta - t_preva[unconda])
        max_step = ta * 10
        tmpa = ta.clone()
        ta = t[active_backtrack] = batched_cubic_interpolate(
            t_preva[unconda],
            f_preva[unconda],
            gtd_preva[unconda],
            ta,
            f_newa,
            gtd_newa,
            bounds=(min_step, max_step),
        )

        # next step
        t_preva = tmpa
        f_preva = f_newa.clone(memory_format=torch.contiguous_format)
        g_preva = g_newa.clone(memory_format=torch.contiguous_format)
        gtd_preva = gtd_newa.clone(memory_format=torch.contiguous_format)
        g_newa, f_newa = (
            g_new[active_backtrack],
            f_new[active_backtrack],
        ) = active_grad_and_obj(
            x[active_backtrack] + ta[:, None] * d[active_backtrack],
            active_backtrack,
        )
        ls_func_evals[active_backtrack] += 1
        gtd_newa = gtd_new[active_backtrack] = (
            g_new[active_backtrack] * d[active_backtrack]
        ).sum(1)
        ls_iter[active_backtrack] += 1

    # reached max number of iterations?
    reached_max = ls_iter == max_ls
    if reached_max.any():
        bracket_l[reached_max] = 0
        bracket_u[reached_max] = t[reached_max]
        bracket_fl[reached_max] = f[reached_max]
        bracket_fu[reached_max] = f_new[reached_max]
        bracket_gl[reached_max] = g[reached_max]
        bracket_gu[reached_max] = g_new[reached_max]
        # these aren't set here for some reason yet unclear
        # bracket_gtdl[reached_max] = gtd_prev[reached_max]
        # bracket_gtdu[reached_max] = gtd_new[reached_max]

    # stack the brackets so that the high_pos / low_pos logic works
    bracket = torch.column_stack((bracket_l, bracket_u))
    bracket_f = torch.column_stack((bracket_fl, bracket_fu))
    bracket_g = torch.column_stack((bracket_gl[:, None], bracket_gu[:, None]))
    bracket_gtd = torch.column_stack((bracket_gtdl, bracket_gtdu))
    gtda = gtd
    # just making sure we don't touch these again
    del (
        bracket_l,
        bracket_u,
        bracket_fl,
        bracket_fu,
        bracket_gl,
        bracket_gu,
        bracket_gtdl,
        bracket_gtdu,
    )

    # zoom phase: we now have a point satisfying the criteria, or
    # a bracket around it. We refine the bracket until we find the
    # exact point satisfying the criteria
    insuf_progress = torch.zeros(B, dtype=torch.bool, device=t.device)
    active_zoom = torch.arange(B, device=t.device)
    # find high and low points in bracket
    high_pos_ac = high_pos = (torch.diff(bracket_f, dim=1)[:, 0] >= 0).to(
        torch.int
    )
    low_pos_ac = low_pos = 1 - high_pos

    while not done.all() and (ls_iter[active_zoom] < max_ls).any():
        # line-search bracket is so small
        dbracket = torch.diff(bracket[active_zoom], dim=1).abs()[:, 0]
        smallbracket = dbracket * d_norm[active_zoom] < tolerance_change
        smalliter = ls_iter[active_zoom] < max_ls
        newmask = smalliter & ~smallbracket
        active_zoom = active_zoom[newmask]
        if not active_zoom.numel():
            break

        bracket_la, bracket_ua = bracket[active_zoom].T
        high_pos_ac = high_pos_ac[newmask]
        low_pos_ac = low_pos_ac[newmask]
        gtda = gtda[newmask]

        # compute new trial value
        ta = t[active_zoom] = batched_cubic_interpolate(
            bracket_la,
            bracket_f[active_zoom, 0],
            bracket_gtd[active_zoom, 0],
            bracket_ua,
            bracket_f[active_zoom, 1],
            bracket_gtd[active_zoom, 1],
        )

        # test that we are making sufficient progress:
        # in case `t` is so close to boundary, we mark that we are making
        # insufficient progress, and if
        #   + we have made insufficient progress in the last step, or
        #   + `t` is at one of the boundary,
        # we will move `t` to a position which is `0.1 * len(bracket)`
        # away from the nearest boundary point.
        bracket_max = torch.max(bracket_la, bracket_ua)
        bracket_min = torch.min(bracket_la, bracket_ua)
        eps = 0.1 * (bracket_max - bracket_min)
        dt_max = bracket_max - ta
        dt_min = ta - bracket_min

        cond1 = torch.min(dt_max, dt_min) < eps
        cond2 = (
            insuf_progress[active_zoom]
            | (ta >= bracket_max)
            | (ta <= bracket_min)
        )
        cond3 = dt_max.abs() < dt_min.abs()
        conda = cond1 & cond2 & cond3
        condb = cond1 & cond2 & ~cond3
        t[active_zoom[conda]] = bracket_max[conda] - eps[conda]
        t[active_zoom[condb]] = bracket_min[condb] + eps[condb]
        insuf_progress[active_zoom] = cond1 & ~cond2
        ta = t[active_zoom]

        # Evaluate new point
        g_newa, f_newa = (
            g_new[active_zoom],
            f_new[active_zoom],
        ) = active_grad_and_obj(
            x[active_zoom] + ta[:, None] * d[active_zoom], active_zoom
        )
        ls_func_evals[active_zoom] += 1
        gtd_newa = gtd_new[active_zoom] = (g_newa * d[active_zoom]).sum(dim=1)
        ls_iter[active_zoom] += 1

        # condx: Armijo condition not satisfied or not lower than lowest point
        cond1 = (f_newa > f[active_zoom] + c1 * ta * gtda) | (
            f_newa >= bracket_f[active_zoom, low_pos_ac]
        )
        # condy: wolfe conditions not satisfied
        cond2 = gtd_newa.abs() <= -c2 * gtda
        # condz: old high becomes new low
        cond3 = (
            gtd_newa
            * (
                bracket[active_zoom, high_pos_ac]
                - bracket[active_zoom, low_pos_ac]
            )
            >= 0
        )
        # condw: new point becomes new low

        condx = cond1
        condy = ~cond1 & cond2
        condz = ~cond1 & ~cond2 & cond3
        condw = ~cond1

        # condx
        indsx = active_zoom[condx]
        high_pos_aca = high_pos_ac[condx]
        bracket[indsx, high_pos_aca] = t[indsx]
        bracket_f[indsx, high_pos_aca] = f_new[indsx]
        bracket_g[indsx, high_pos_aca] = g_new[indsx]
        bracket_gtd[indsx, high_pos_aca] = gtd_new[indsx]
        high_pos_ac[condx] = high_pos[indsx] = (
            torch.diff(bracket_f[indsx], dim=1)[:, 0] >= 0
        ).to(torch.int)
        low_pos_ac[condx] = low_pos[indsx] = 1 - high_pos_ac[condx]

        # condz
        indsz = active_zoom[condz]
        high_pos_acc = high_pos_ac[condz]
        low_pos_acc = low_pos_ac[condz]
        bracket[indsz, high_pos_acc] = bracket[indsz, low_pos_acc]
        bracket_f[indsz, high_pos_acc] = bracket_f[indsz, low_pos_acc]
        bracket_g[indsz, high_pos_acc] = bracket_g[indsz, low_pos_acc]
        bracket_gtd[indsz, high_pos_acc] = bracket_gtd[indsz, low_pos_acc]

        # condw
        indsw = active_zoom[condw]
        low_pos_acd = low_pos_ac[condw]
        bracket[indsw, low_pos_acd] = ta[condw]
        bracket_f[indsw, low_pos_acd] = f_newa[condw]
        bracket_g[indsw, low_pos_acd] = g_newa[condw]
        bracket_gtd[indsw, low_pos_acd] = gtd_newa[condw]

        # condy last bc it updates active set
        done[active_zoom[condy]] = True
        active_zoom = active_zoom[~condy]

        bracket_la, bracket_ua = bracket[active_zoom].T
        high_pos_ac = high_pos_ac[~condy]
        low_pos_ac = low_pos_ac[~condy]
        gtda = gtda[~condy]

    # return stuff
    t = bracket[torch.arange(B), low_pos]
    f_new = bracket_f[torch.arange(B), low_pos]
    g_new = bracket_g[torch.arange(B), low_pos]
    return f_new, g_new, t, ls_func_evals
