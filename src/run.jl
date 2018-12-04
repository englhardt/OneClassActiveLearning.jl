
function init_from_experiment(experiment, data, labels, res)
    debug(LOGGER, "[INIT] Start initialization of experiment.")
    pools = copy(experiment[:param][:initial_pools])
    split_strategy = experiment[:split_strategy]

    # model
    train_data, train_pools, _ = get_train(split_strategy, data, pools)
    debug(LOGGER, "[INIT] Initializing model '$(experiment[:model])' with $(format_observations(train_data)) observations.")
    model = instantiate(eval(experiment[:model][:type]), train_data, train_pools, experiment[:model][:param])
    initialize!(model, eval(experiment[:model][:init_strategy]))
    set_model_fitted!(res, model)
    set_adjust_K!(model, experiment[:param][:adjust_K])
    solver = experiment[:param][:solver]
    debug(LOGGER, "[INIT] Model solver for this experiment is '$(typeof(solver))'.")

    # query strategy
    debug(LOGGER, "[INIT] Initializing QS '$(experiment[:query_strategy][:type])'.")
    query_data, _, _ = get_query(split_strategy, data, pools)
    qs = QueryStrategies.initialize_qs(eval(experiment[:query_strategy][:type]), model, query_data, experiment[:query_strategy][:param])

    debug(LOGGER, "[INIT] Initializing oracle.")
    if isa(experiment[:oracle], Oracle)
        oracle = experiment[:oracle]
    else
        oracle = initialize_oracle(eval(experiment[:oracle][:type]), data, labels, experiment[:oracle][:param])
    end

    info(LOGGER, "[INIT] Initialization done.")
    return (model, pools, solver, qs, split_strategy, oracle)
end

function check_active_learn_args(data, labels)
    MLLabelUtils.islabelenc(labels, OneClassActiveLearning.LABEL_ENCODING) || throw(ArgumentError("Argument labels is in the wrong encoding."))
    return size(data, 2) == length(labels) || throw(ArgumentError("Number of observations ($(size(data,2))) does not equal number of labels $(length(labels))."))
end

function active_learn(experiment::Dict{Symbol, Any})
    data, labels = OneClassActiveLearning.load_data(experiment[:data_file])
    return active_learn(experiment, data, labels)
end

function active_learn(experiment::Dict{Symbol, Any}, data::Array{T, 2}, labels::Vector{Symbol}) where T <: Real
    check_active_learn_args(data, labels)
    log_experiment_info(experiment)

    res = Result(experiment)
    set_worker_info!(res)
    set_data_stats!(res, data, experiment[:split_strategy])

    model, pools, solver, qs, split_strategy, oracle = try
        init_from_experiment(experiment, data, labels, res)
    catch e
        warn(LOGGER, e)
        if isa(e, KDEException)
            res.status[:exit_code] = :kde_error
        elseif isa(e, MissingLabelTypeException)
            res.status[:exit_code] = :missing_label_type
        else
            throw(e)
        end
        return res
    end

    debug(LOGGER, "Start active learning cycle with $(experiment[:param][:num_al_iterations]) queries.")
    for i in 0:experiment[:param][:num_al_iterations]
        info(LOGGER, "Iteration $(i)")
        debug(LOGGER, "Memory consumption $(round(Int, Sys.free_memory() / 2^20)) MB / $(round(Int, Sys.total_memory() / 2^20)) MB")

        train_data, train_pools, _ = get_train(split_strategy, data, pools)
        time_set_data = @elapsed set_data!(model, train_data)
        set_pools!(model, labelmap(train_pools))

        debug(LOGGER, "[FIT] Start fitting model on $(format_observations(train_data)) observations.")
        # Workaround: redirect solver output
        stdout_orig, stderr_orig = stdout, stderr
        redirect_stdout(); redirect_stderr()
        status, time_fit, mem_fit = @timed SVDD.fit!(model, solver)
        redirect_stdout(stdout_orig); redirect_stderr(stderr_orig)
        debug(LOGGER, "[FIT] Fitting done ($(time_fit) s, $(format_bytes(mem_fit))).")

        @trace res.al_history i time_fit mem_fit time_set_data
        if status !== JuMP.MathOptInterface.Success
            warn(LOGGER, "Not solved to optimality. Solver status: $status.")
            res.status[:exit_code] = :solver_error
            return res
        end

        test_data, _, test_indices = get_test(split_strategy, data, pools)
        debug(LOGGER, "[TEST] Testing by predicting $(format_observations(test_data)) observations.")
        predictions = SVDD.predict(model, test_data)
        push_evaluation!(res.al_history, i, predictions, labels[test_indices])
        debug(LOGGER, "[TEST] Testing done.")

        if i < experiment[:param][:num_al_iterations]
            query_data, query_pools, query_indices = try
                get_query(split_strategy, data, pools)
            catch e
                warn(LOGGER, e)
                if isa(e, KDEException)
                     res.status[:exit_code] = :kde_error
                elseif isa(e, MissingLabelTypeException)
                    res.status[:exit_code] = :missing_label_type
                else
                    res.status[:exit_code] = :unknown_qs_error
                end
                return res
            end
            if :U ∉ query_pools && !isa(qs, QuerySynthesisStrategy)
                info(LOGGER, "Aborting '$(experiment[:hash])' after $(i) iterations because no more unlabeled observations are left.")
                al_summarize!(res)
                res.status[:exit_code] = :early_abort
                return res
            end
            debug(LOGGER, "[QS] Starting query strategy on $(format_observations(query_data)) observations.")
            if i == 0
                query, time_qs, mem_qs = @timed get_query_object_helper(qs, query_data, query_pools, query_indices)
            else
                query, time_qs, mem_qs = @timed get_query_object_helper(qs, query_data, query_pools, query_indices, values(res.al_history, :query_history))
            end
            debug(LOGGER, "[QS] Query strategy finished ($(time_qs) s, $(format_bytes(mem_qs))).")
            query_label = ask_oracle(oracle, query)
            data = update_data_and_pools!(qs, data, labels, pools, split_strategy, query, query_label)
            push_query!(res.al_history, i, query, query_label, time_qs, mem_qs)
            isa(query, Int) ? debug(LOGGER, "[QS] Query(id = $(query), label = $(query_label))") :
                             debug(LOGGER, "[QS] Query(label = $(query_label))")
            debug(LOGGER, "[QS] Query strategy done.")


        end
        debug(LOGGER, "Finished iteration $(i).")
    end
    debug(LOGGER, "Finished active learning cycle.")

    al_summarize!(res)
    debug(LOGGER, "Summary done.")
    info(LOGGER, "Finished experiment '$(experiment[:hash])'.")
    res.status[:exit_code] = :success
    return res
end

function get_query_object_helper(qs::Q,
                                 query_data::Array{<:Real, 2},
                                 query_pools::Vector{Symbol},
                                 query_indices::Vector{Int},
                                 history::Vector{Int}=Int[])::Int where Q <: Union{PoolQs, SubspaceQs}
    return get_query_object(qs, query_data, query_pools, query_indices, history)
end

function get_query_object_helper(qs::QuerySynthesisStrategy,
                                 query_data::Array{T, 2},
                                 query_pools::Vector{Symbol},
                                 query_indices::Vector{Int},
                                 history::Vector{Array{T, 2}}=Vector{Array{T, 2}}())::Array{T, 2} where T <: Real
    return get_query_object(qs, query_data, query_pools, history)
end

function update_data_and_pools!(qs::SubspaceQs, data, labels, pools, split_strategy, query, query_label)
    pools[query] = query_label == :inlier ? :Lin : :Lout
    return data
end

function update_data_and_pools!(qs::PoolQs, data, labels, pools, split_strategy, query, query_label)
    pools[query] = query_label == :inlier ? :Lin : :Lout
    return data
end

function update_data_and_pools!(qs::QuerySynthesisStrategy, data, labels, pools, split_strategy, query, query_label)
    data = hcat(data, query)
    push!(labels, query_label)
    push!(pools, query_label == :inlier ? :Lin : :Lout)
    push!(split_strategy.train, true)
    push!(split_strategy.test, false)
    return data
end

function push_query!(al_history::MVHistory, i, query, query_label, time_qs, mem_qs)
    push!(al_history, :query_history, i, query)
    @trace al_history i query_label time_qs mem_qs
    return nothing
end

function push_evaluation_cm!(al_history, i, cm)
    push!(al_history, :cm, i, cm)
    for e in [:cohens_kappa, :matthews_corr, :f1_score, :tpr, :fpr]
        push!(al_history, e, i, eval(e)(cm))
    end
end

function push_evaluation!(al_history::MVHistory, i, predictions::Vector{Vector{Float64}}, labels)
    cm = ConfusionMatrix(SVDD.classify(predictions, Val(:Global)), labels)
    push_evaluation_cm!(al_history, i, cm)
    return nothing
end

function push_evaluation!(al_history::MVHistory, i, predictions, labels)
    cm = ConfusionMatrix(SVDD.classify.(predictions), labels)
    push_evaluation_cm!(al_history, i, cm)
    push!(al_history, :auc, i, roc_auc(predictions, labels))
    for k in [0.01, 0.02, 0.05, 0.1, 0.2]
        auc_fpr = OneClassActiveLearning.roc_auc(predictions, labels, fpr = k)
        auc_fpr_normalized = OneClassActiveLearning.roc_auc(predictions, labels, fpr = k, normalize = true)
        push!(al_history, Symbol("auc_fpr_$(k)"), i, auc_fpr)
        push!(al_history, Symbol("auc_fpr_normalized_$(k)"), i, auc_fpr_normalized)
    end
    return nothing
end
