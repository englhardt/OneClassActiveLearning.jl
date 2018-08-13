
function init_from_experiment(experiment, data, res)
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

    info(LOGGER, "[INIT] Initialization done.")
    return (model, pools, solver, qs, split_strategy)
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

    model, pools, solver, qs, split_strategy = try
        init_from_experiment(experiment, data, res)
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
        STDOUT_orig, STDERR_orig = STDOUT, STDERR
        redirect_stdout(open("/dev/null", "w")); redirect_stderr(open("/dev/null", "w"))
        status, time_fit, mem_fit = @timed fit!(model, solver)
        redirect_stdout(STDOUT_orig); redirect_stderr(STDERR_orig)
        debug(LOGGER, "[FIT] Fitting done ($(time_fit) s, $(format_bytes(mem_fit))).")

        @trace res.al_history i time_fit mem_fit time_set_data
        if status != :Optimal
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
            if :U ∉ query_pools
                info(LOGGER, "Aborting '$(experiment[:hash])' after $(i) iterations because no more unlabeled observations are left.")
                al_summarize!(res)
                res.status[:exit_code] = :early_abort
                return res
            end
            debug(LOGGER, "[QS] Starting query strategy on $(format_observations(query_data)) observations.")
            if i == 0
                query_id, time_qs, mem_qs = @timed get_query_object(qs, query_data, query_pools, query_indices, Int[])
            else
                query_id, time_qs, mem_qs = @timed get_query_object(qs, query_data, query_pools, query_indices, values(res.al_history, :query_history))
            end
            debug(LOGGER, "[QS] Query strategy scoring done ($(time_qs) s, $(format_bytes(mem_qs))).")
            update_pools!(pools, query_id, labels)
            push_query!(res.al_history, i, query_id, labels[query_id], time_qs, mem_qs)
            debug(LOGGER, "[QS] Query(id = $(query_id), label = $(labels[query_id]))")
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

function update_pools!(pools, query_id, labels)
    pools[query_id] = labels[query_id] == :inlier ? :Lin : :Lout
    return nothing
end

"""
get_query_object(qs::QueryStrategy, data::Array{T, 2}, pools::Vector{Symbol}, global_indices::Vector{Int}, history::Vector{Int})

# Arguments
- `query_data`: Subset of the full data set
- `pools`: Labels for `query_data`
- `global_indices`: Indices of the observations in `query_data` relative to the full data set.
"""
function get_query_object(qs::QueryStrategy, query_data::Array{T, 2}, pools::Vector{Symbol}, global_indices::Vector{Int}, history::Vector{Int})::Int where T <: Real
    pool_map = labelmap(pools)
    haskey(pool_map, :U) || throw(ArgumentError("No more points that are unlabeled."))
    scores = qs_score(qs, query_data, pool_map)
    @assert length(scores) == size(query_data, 2)
    candidates = [i for i in pool_map[:U] if global_indices[i] ∉ history]
    debug(LOGGER, "[QS] Selecting from $(format_number(length(candidates))) candidates.")
    local_query_index = candidates[indmax(scores[candidates])]
    return global_indices[local_query_index]
end

function push_query!(al_history::MVHistory, i, query_id, query_label, time_qs, mem_qs)
    push!(al_history, :query_history, i, query_id)
    @trace al_history i query_label time_qs mem_qs
    return nothing
end

function push_evaluation!(al_history::MVHistory, i, predictions, labels)
    cm = ConfusionMatrix(SVDD.classify.(predictions), labels)
    push!(al_history, :cm, i, cm)
    for e in [:cohens_kappa, :matthews_corr, :f1_score, :tpr, :fpr]
        push!(al_history, e, i, eval(e)(cm))
    end
    push!(al_history, :auc, i, roc_auc(predictions, labels))
    for k in [0.01, 0.02, 0.05, 0.1, 0.2]
        auc_fpr = OneClassActiveLearning.roc_auc(predictions, labels, fpr = k)
        auc_fpr_normalized = OneClassActiveLearning.roc_auc(predictions, labels, fpr = k, normalize = true)
        push!(al_history, Symbol("auc_fpr_$(k)"), i, auc_fpr)
        push!(al_history, Symbol("auc_fpr_normalized_$(k)"), i, auc_fpr_normalized)
    end
    return nothing
end
