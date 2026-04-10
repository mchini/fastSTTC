function result = sttc(spike_matrix, dt_values)
% STTC  Pairwise STTC for all units at one or more Δt values.
%
%   Strategy
%   --------
%   For each Δt:
%     1. Build the tiled matrix with the cumsum trick  — O(N·T)
%     2. Compute all N² coincidence counts in one matrix multiply:
%            C = spike_matrix * tiled'         size (N, N)
%     3. Derive PA, TA and apply the STTC formula element-wise.
%
%   The only loop is over dt_values (typically ≤ 20 iterations).
%
%   Parameters
%   ----------
%   spike_matrix : (N, T) binary array (uint8 or logical)
%       N units × T time bins. Each 1 indicates a spike in that bin.
%   dt_values : scalar int or vector of ints
%       Half-window(s) in bins. Pass a scalar for one Δt; pass a vector
%       for several Δt values computed in one call.
%
%   Returns
%   -------
%   result : single array
%       - If dt_values is scalar: (N, N). Diagonal entries are NaN.
%       - If dt_values is a vector: (N, N, n_dt).
%         result(:, :, k) is the STTC matrix at dt_values(k).
%
%   Examples
%   --------
%   spike_matrix = bin_spike_times(spike_times, 300000);
%   S = sttc(spike_matrix, 50);           % single Δt → (N, N)
%   S = sttc(spike_matrix, [10 25 50]);   % three Δt  → (N, N, 3)

scalar_input = isscalar(dt_values);
dt_list      = dt_values(:)';   % row vector
n_dt         = numel(dt_list);

sf   = single(spike_matrix);
[N, ~] = size(sf);
n_sp = sum(sf, 2);              % (N, 1) spike counts

out = nan(N, N, n_dt, 'single');

for k = 1:n_dt
    dt    = dt_list(k);
    tiled = tile_spikes(spike_matrix, dt);   % (N, T) single
    TA    = mean(tiled, 2);                  % (N, 1)
    C     = sf * tiled';                     % (N, N) — all coincidences

    % PA(a,b) = fraction of a's spikes in tiles of b
    PA = C ./ n_sp;                          % (N, N); NaN where n_sp == 0

    % PB(a,b) = PA(b,a) = fraction of b's spikes in tiles of a
    d1 = 1.0 - PA   .* TA';   % TA' broadcasts along rows → TA(b) per column b
    d2 = 1.0 - PA'  .* TA;    % TA  broadcasts along cols → TA(a) per row    a

    t1 = (PA  - TA') ./ d1;
    t2 = (PA' - TA)  ./ d2;
    t1(abs(d1) <= 1e-10) = NaN;
    t2(abs(d2) <= 1e-10) = NaN;

    mat_k = 0.5 * (t1 + t2);
    mat_k(1:N+1:end) = NaN;    % diagonal → NaN
    out(:, :, k) = mat_k;
end

if scalar_input
    result = out(:, :, 1);
else
    result = out;
end
end
