function spike_times_list = generate_spike_trains(N, T_ms, rate_hz, correlation, seed)
% GENERATE_SPIKE_TRAINS  Generate synthetic Poisson spike trains, optionally correlated.
%
%   Useful for testing and for the examples notebook.
%
%   Parameters
%   ----------
%   N           : int — number of units
%   T_ms        : float — recording duration in ms
%   rate_hz     : float — mean firing rate in Hz (default 5)
%   correlation : float in [0, 1] — fraction of spikes that are shared
%                 across all units via a common Poisson process.
%                 0 = independent, 1 = perfectly correlated.
%   seed        : int or [] — random seed (default [])
%
%   Returns
%   -------
%   spike_times_list : cell array of N sorted single arrays (spike times in ms)

if nargin < 3 || isempty(rate_hz),     rate_hz     = 5.0; end
if nargin < 4 || isempty(correlation), correlation = 0.0; end
if nargin < 5 || isempty(seed),        seed        = [];   end

if ~isempty(seed)
    rng(seed);
end

T_s              = T_ms / 1000.0;
spike_times_list = cell(N, 1);

% shared events (drive correlations)
if correlation > 0
    n_shared = poissrnd(rate_hz * T_s * correlation);
    shared   = sort(single(rand(1, n_shared) * T_ms));
else
    shared = single([]);
end

indep_rate = rate_hz * (1.0 - correlation);

for i = 1:N
    n_indep = poissrnd(indep_rate * T_s);
    indep   = single(rand(1, n_indep) * T_ms);
    spikes  = sort([shared, indep]);
    spike_times_list{i} = spikes;
end
end
