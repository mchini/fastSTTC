function spike_matrix = bin_spike_times(spike_times_list, T_ms, bin_size_ms)
% BIN_SPIKE_TIMES  Convert spike times to a binary spike matrix suitable for STTC.
%
%   Parameters
%   ----------
%   spike_times_list : cell array of 1-D arrays
%       Each element contains the spike times (in ms) for one unit.
%   T_ms             : float — total recording duration in ms
%   bin_size_ms      : float — width of each time bin in ms (default 1.0)
%
%   Returns
%   -------
%   spike_matrix : (N, T_bins) uint8 array
%       1 where a spike occurred, 0 elsewhere.
%       If two spikes fall in the same bin only one is counted.
%
%   Examples
%   --------
%   spike_matrix = bin_spike_times(spike_times, 300000);
%   % spike_matrix has size (N, 300000)

if nargin < 3, bin_size_ms = 1.0; end

T_bins       = ceil(T_ms / bin_size_ms);
N            = numel(spike_times_list);
spike_matrix = zeros(N, T_bins, 'uint8');

for i = 1:N
    spikes = double(spike_times_list{i});
    if isempty(spikes), continue; end
    idx = floor(spikes / bin_size_ms) + 1;   % convert to 1-indexed bins
    idx = idx(idx >= 1 & idx <= T_bins);
    spike_matrix(i, idx) = 1;
end
end
