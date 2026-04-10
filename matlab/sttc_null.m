function null = sttc_null(spike_matrix, dt, n_shifts, shuffle, seed)
% STTC_NULL  Null distribution for pairwise STTC via circular shifts or shuffles.
%
%   For each of n_shifts iterations the spike trains are either circularly
%   shifted by a random amount or randomly shuffled (destroying temporal
%   structure while preserving spike counts), and the pairwise STTC is
%   recomputed.  The resulting distribution approximates what STTC values
%   look like when there is no genuine correlation between units.
%
%   Optimisation: the tiled version of the original signal and TA are
%   constant across all shifts → computed once before the loop.  Each
%   iteration only tiles the shifted signal (one O(N·T) call instead of two).
%
%   Parameters
%   ----------
%   spike_matrix : (N, T) binary array (uint8 or logical)
%   dt           : int — half-window in bins
%   n_shifts     : int — number of surrogate matrices (default 200)
%   shuffle      : logical — if true, randomly permute time bins instead of
%                  circularly shifting.  Circular shift (default) is preferred
%                  because it preserves the autocorrelation structure of each
%                  unit; shuffle destroys it.
%   seed         : int or [] — random seed for reproducibility (default [])
%
%   Returns
%   -------
%   null : (n_shifts, N, N) single array
%       null(s, i, j) is the STTC between unit i and the surrogate of
%       unit j at shift s.  Diagonal entries are NaN.
%
%   Notes
%   -----
%   Circular shifts are drawn uniformly from
%   [max(dt+1, floor(T/10)),  T - max(dt+1, floor(T/10)) - 1]
%   to ensure each surrogate is genuinely decorrelated from the original.

if nargin < 3 || isempty(n_shifts), n_shifts = 200;  end
if nargin < 4 || isempty(shuffle),  shuffle  = false; end
if nargin < 5 || isempty(seed),     seed     = [];    end

if ~isempty(seed)
    rng(seed);
end

sf   = single(spike_matrix);
[N, T] = size(sf);
n_sp = sum(sf, 2);   % (N, 1) — constant across surrogates

% hoist: tile original once
tiled_orig = single(tile_spikes(spike_matrix, dt));   % (N, T)
TA         = mean(tiled_orig, 2);                     % (N, 1)

% shift values: random, well-separated from both boundaries
min_shift = max(dt + 1, floor(T / 10));
shifts    = randi([min_shift, T - min_shift - 1], 1, n_shifts);

null = nan(n_shifts, N, N, 'single');

for s = 1:n_shifts
    if shuffle
        shifted = spike_matrix(:, randperm(T));
    else
        shifted = circshift(spike_matrix, shifts(s), 2);
    end

    sf_s    = single(shifted);
    tiled_s = single(tile_spikes(shifted, dt));   % 1 tile call per iter
    TB      = mean(tiled_s, 2);                   % (N, 1)

    % C_AB(i,j) = #{spikes of orig[i]    within tiles of shifted[j]}
    % C_BA(i,j) = #{spikes of shifted[j] within tiles of orig[i]}
    C_AB = sf          * tiled_s';   % (N, N) — orig spikes  × shifted tiles
    C_BA = tiled_orig  * sf_s';      % (N, N) — orig tiles   × shifted spikes
    %      ↑ reuses the hoisted tiled_orig

    PA = C_AB ./ n_sp;    % (N, N); PA(i,j) = C_AB(i,j) / n_sp(i)
    PB = C_BA ./ n_sp';   % (N, N); PB(i,j) = C_BA(i,j) / n_sp(j)

    d1 = 1 - PA .* TB';
    d2 = 1 - PB .* TA;
    t1 = (PA - TB') ./ d1;
    t2 = (PB - TA)  ./ d2;
    t1(abs(d1) <= 1e-10) = NaN;
    t2(abs(d2) <= 1e-10) = NaN;

    mat_s = 0.5 * (t1 + t2);
    mat_s(1:N+1:end) = NaN;
    null(s, :, :) = mat_s;
end
end
