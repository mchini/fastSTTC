function z = zscore_sttc(spike_matrix, dt, n_shifts, shuffle, seed)
% ZSCORE_STTC  Z-scored pairwise STTC.
%
%   Computes the observed STTC matrix and a null distribution (via
%   sttc_null), then returns:
%
%       z(i, j) = (STTC(i,j) - mean(null(:, i, j))) / std(null(:, i, j))
%
%   A z-score above ~2–3 indicates that units i and j co-fire significantly
%   more than expected by chance given their individual firing rates.
%
%   Parameters
%   ----------
%   spike_matrix : (N, T) binary array (uint8 or logical)
%   dt           : int — half-window in bins
%   n_shifts     : int — surrogate count (default 200; use ≥ 500 for publication)
%   shuffle      : logical — see sttc_null (default false)
%   seed         : int or [] — for reproducibility (default [])
%
%   Returns
%   -------
%   z : (N, N) single array.  Diagonal entries are NaN.

if nargin < 3 || isempty(n_shifts), n_shifts = 200;  end
if nargin < 4 || isempty(shuffle),  shuffle  = false; end
if nargin < 5 || isempty(seed),     seed     = [];    end

observed  = sttc(spike_matrix, dt);
null      = sttc_null(spike_matrix, dt, n_shifts, shuffle, seed);

null_mean = squeeze(mean(null, 1, 'omitnan'));   % (N, N)
null_std  = squeeze(std(null,  0, 1, 'omitnan')); % (N, N)

z = single((observed - null_mean) ./ null_std);
end
