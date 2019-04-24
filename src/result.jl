
mutable struct DataStats
    num_observations::Int
    num_dimensions::Int
    train_fraction::Float64
    test_fraction::Float64
    train_indices::Vector{Int}
    test_indices::Vector{Int}
    DataStats() = new(0, 0, 0.0, 0.0, Int[], Int[])
    DataStats(num_observations::Int, num_dimensions::Int, train_fraction::Float64, test_fraction::Float64,
              train_indices::Vector{Any}, test_indices::Vector{Any}) = new(num_observations, num_dimensions, train_fraction, test_fraction,
                                                                           convert(Vector{Int}, train_indices), convert(Vector{Int}, test_indices))
    DataStats(num_observations::Int, num_dimensions::Int, train_fraction::Float64, test_fraction::Float64,
              train_indices::Vector{Int}, test_indices::Vector{Int}) = new(num_observations, num_dimensions, train_fraction, test_fraction,
                                                                           train_indices, test_indices)
end

struct Result
    id
    experiment::Dict
    worker_info::Dict{Symbol, String}
    data_stats::DataStats
    al_history::ValueHistories.MVHistory
    al_summary::Dict
    status::Dict{Symbol, Symbol}
    Result(experiment) = new(experiment[:hash],
                     experiment,
                     Dict{Symbol, String}(),
                     DataStats(),
                     ValueHistories.MVHistory(),
                     Dict{Symbol, Dict{Symbol, Any}}(),
                     Dict(:exit_code => :initialized))
    Result(id, experiment, worker_info, al_history, al_summary) = new(id, experiment, worker_info, DataStats(),
                                                                      al_history, al_summary, Dict(:exit_code => :initialized))
    Result(id, experiment, worker_info, data_stats, al_history, al_summary) = new(id, experiment, worker_info, data_stats, al_history, al_summary,
                                                                                  Dict(:exit_code => :initialized))
    Result(id, experiment, worker_info, data_stats, al_history, al_summary, status) = new(id, experiment, worker_info, data_stats, al_history, al_summary, status)
end

function set_data_stats!(res::Result, x::Array{T, 2}, ds::DataSplits) where T <: Real
    res.data_stats.num_observations = size(x, 2)
    res.data_stats.num_dimensions = size(x, 1)
    res.data_stats.train_fraction = sum(ds.train) / res.data_stats.num_observations
    res.data_stats.test_fraction = sum(ds.test) / res.data_stats.num_observations
    res.data_stats.train_indices = findall(ds.train)
    res.data_stats.test_indices = findall(ds.test)
    return nothing
end

function set_model_fitted!(res, model)
    res.experiment[:model][:fitted] = Dict(:kernel => SVDD.get_kernel(model),
                              :model_params => SVDD.get_model_params(model))
    return nothing
end

function get_worker_info(;debug = false)
    worker_info = Dict{Symbol, String}()
    worker_info[:julia_version] = string(VERSION)
    debug && (worker_info[:installed_packages] = join( ["$k [$v]" for (k,v) in Pkg.installed()], ','))
    dir = @__DIR__
    cmd = Sys.iswindows() ? `cmd /c git -C $(dir) rev-parse HEAD` : `git -C $(@__DIR__) rev-parse HEAD`
    worker_info[:git_commit] = strip(string(read(cmd, String)))
    worker_info[:utc_time] = string(Dates.now(Dates.UTC))
    worker_info[:hostname] = gethostname()
    return worker_info
end

function set_worker_info!(res::Result)
    merge!(res.worker_info, get_worker_info())
    return nothing
end

function al_summarize!(res::Result)
    for e in keys(res.al_history)
        # Skip calculation of summary statistics for some metrics
        if e ∈ [:cm, :time_fit, :mem_fit, :time_qs, :mem_qs, :query_labels,
            :query_history, :runtime, :time_set_data]
            continue
        end
        res.al_summary[e] = Dict()
        scores = values(res.al_history, e)
        score_changes = scores[2:end] - scores[1:end-1]
        res.al_summary[e][:start_quality] = scores[1]
        res.al_summary[e][:end_quality] = scores[end]
        res.al_summary[e][:maximum] = maximum(scores)
        res.al_summary[e][:ramp_up] = scores[2:end] .- scores[1]
        res.al_summary[e][:quality_range] = scores[end] .- scores[1]
        res.al_summary[e][:total_quality_range] = maximum(scores) - minimum(scores)
        res.al_summary[e][:average_end_quality] = [mean(scores[k:end]) for k in 1:length(scores)]
        res.al_summary[e][:average_quality_change] = mean(score_changes)
        if res.al_summary[e][:quality_range] > 0
            res.al_summary[e][:learning_stability] = [((scores[end] - scores[end-k]) / k) / (res.al_summary[e][:quality_range] / (length(scores) - 1)) for k in 1:(length(scores) - 1)]
        else
            res.al_summary[e][:learning_stability] = [0.0 for k in 1:(length(scores) - 1)]
        end
        res.al_summary[e][:average_gain] = any(score_changes .> 0) ? mean(score_changes[score_changes .> 0]) : 0
        res.al_summary[e][:average_loss] = any(score_changes .< 0) ? mean(score_changes[score_changes .< 0]) : 0
        query_labels = values(res.al_history, :query_labels)
        if :outlier in query_labels
            res.al_summary[e][:ratio_of_outlier_queries] = sum(query_labels .== :outlier) / length(query_labels)
        else
            res.al_summary[e][:ratio_of_outlier_queries] = 0.0
        end
        # Summary statistics from
        # Reyes, O. et al. 2018. Statistical comparisons of active learning strategies over multiple datasets.
        # Knowledge-Based Systems. 145, (2018), 1–14. DOI:https://doi.org/10.1016/j.knosys.2018.01.033
        res.al_summary[e][:aulc] = 0.5 * sum(scores[2:end] + scores[1:end-1])
        score_changes_pos = [scores[i] + scores[i+1] for i in 1:length(scores)-1 if scores[i+1] > scores[i]]
        res.al_summary[e][:reyes_paulc] = isempty(score_changes_pos) ? 0.0 : 0.5 * sum(score_changes_pos)
        score_changes_non_pos = [scores[i] + scores[i+1] for i in 1:length(scores)-1 if scores[i+1] <= scores[i]]
        res.al_summary[e][:reyes_naulc] = isempty(score_changes_non_pos) ? 0.0 : 0.5 * sum(score_changes_non_pos)
        res.al_summary[e][:reyes_tpr] = sum(score_changes[score_changes .> 0])
        res.al_summary[e][:reyes_tnr] = sum(score_changes[score_changes .<= 0])
        res.al_summary[e][:reyes_tp] = res.al_summary[e][:reyes_paulc] * res.al_summary[e][:reyes_tpr] - res.al_summary[e][:reyes_naulc] * res.al_summary[e][:reyes_tnr]
    end
    return nothing
end
