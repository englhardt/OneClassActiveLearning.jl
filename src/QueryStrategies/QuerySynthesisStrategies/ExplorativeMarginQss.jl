mutable struct ExplorativeMarginQss <: HybridQss
    occ::SVDD.SVDDClassifier
    occ_eps::SVDD.SVDDnegEps
    occ_eps_init_strat::SVDD.InitializationStrategyCombined
    solver::JuMP.OptimizerFactory
    optimizer::QuerySynthesisOptimizer
    boundary_shift_agg_func::Symbol
    lambda::Float64
    use_penalty::Bool
    eps
    function ExplorativeMarginQss(occ, data; solver::JuMP.OptimizerFactory, optimizer::QuerySynthesisOptimizer,
                                  boundary_shift_agg_func=:maximum, lambda=1.0, use_penalty=true)
        !isa(occ.kernel_fct, SquaredExponentialKernel) && throw(ArgumentError("Invalid kernel type $(typeof(occ.kernel_fct)). Expected type is a SquaredExponentialKernel."))
        occ_eps = SVDD.SVDDnegEps(data, fill(:U, size(data, 2)))
        occ_params = SVDD.get_model_params(occ)
        !haskey(occ_params, :C) && !haskey(occ_params, :C1) && throw(ArgumentError("Invalid base learner type $(typeof(occ)). Cannot retrieve C parameter for SVDDnegEps."))
        occ_eps_init_strat = SVDD.SimpleCombinedStrategy(SVDD.FixedGammaStrategy(MLKernels.GaussianKernel(occ.kernel_fct.alpha.value.x)),
                                                         SVDD.FixedCStrategy(haskey(occ_params, :C) ? occ_params[:C] : occ_params[:C1]))
        new(occ, occ_eps, occ_eps_init_strat, solver, optimizer, boundary_shift_agg_func, lambda, use_penalty, nothing)
    end
end

function qs_score_function(qs::ExplorativeMarginQss, data::Array{T, 2}, labels::Dict{Symbol, Array{Int, 1}})::Function where T <: Real
    # Init epsilon if not yet set
    if qs.eps === nothing
        data_target, data_outliers = SVDD.generate_binary_data_for_tuning(data)
        qs.eps = max(0.0, qs.lambda * estimate_boundary_shift_epsilon(qs.occ, data_outliers, agg_func=qs.boundary_shift_agg_func))
        SVDD.set_eps!(qs.occ_eps, qs.eps)
    end
    if (qs.occ_eps.state === SVDD.model_created)
        SVDD.initialize!(qs.occ_eps, qs.occ_eps_init_strat)
    end
    SVDD.set_data!(qs.occ_eps, data)
    SVDD.set_pools!(qs.occ_eps, labels)
    # Workaround: redirect solver output
    stdout_orig, stderr_orig = stdout, stderr
    redirect_stdout(); redirect_stderr()
    status = SVDD.fit!(qs.occ_eps, qs.solver)
    redirect_stdout(stdout_orig); redirect_stderr(stderr_orig)
    if status === JuMP.MathOptInterface.ALMOST_LOCALLY_SOLVED
        warn(getlogger(@__MODULE__), "Qss occ_eps not solved to optimality. Solver status: $status.")
    elseif status !== JuMP.MathOptInterface.OPTIMAL
        error(getlogger(@__MODULE__), "Qss occ_eps not solved to optimality. Solver status: $status.")
    end
    oc_scoring(x) = -abs.(SVDD.predict(qs.occ_eps, x))

    # Train penalty binary SVM if outlier labels are available
    if qs.use_penalty && haskey(labels, :Lout)
        labels_binary = fill(:inlier, size(data, 2))
        labels_binary[labels[:Lout]] .= :outlier
        model_binary = LIBSVM.svmtrain(data, labels_binary, gamma=qs.occ_eps.kernel_fct.alpha.value.x, cost=100_000.0)
        penalty(x) = LIBSVM.svmpredict(model_binary, x)[2][1, :]
        return x -> oc_scoring(x) .- max.(zeros(size(x, 2)), -penalty(x))
    end
    return oc_scoring
end
